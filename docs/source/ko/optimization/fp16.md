<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 메모리와 속도

메모리 또는 속도에 대해 🤗 Diffusers *추론*을 최적화하기 위한 몇 가지 기술과 아이디어를 제시합니다.
일반적으로, memory-efficient attention을 위해 [xFormers](https://github.com/facebookresearch/xformers) 사용을 추천하기 때문에, 추천하는 [설치 방법](xformers)을 보고 설치해 보세요.

다음 설정이 성능과 메모리에 미치는 영향에 대해 설명합니다.

|                  | 지연시간  | 속도 향상 |
| ---------------- | ------- | ------- |
| 별도 설정 없음      | 9.50s   | x1      |
| cuDNN auto-tuner | 9.37s   | x1.01   |
| fp16             | 3.61s   | x2.63   |
| Channels Last 메모리 형식     | 3.30s   | x2.88   |
| traced UNet      | 3.21s   | x2.96   |
| memory-efficient attention | 2.63s  | x3.61   |

<em>
   NVIDIA TITAN RTX에서 50 DDIM 스텝의 "a photo of an astronaut riding a horse on mars" 프롬프트로 512x512 크기의 단일 이미지를 생성하였습니다.
</em>

## cuDNN auto-tuner 활성화하기

[NVIDIA cuDNN](https://developer.nvidia.com/cudnn)은 컨볼루션을 계산하는 많은 알고리즘을 지원합니다. Autotuner는 짧은 벤치마크를 실행하고 주어진 입력 크기에 대해 주어진 하드웨어에서 최고의 성능을 가진 커널을 선택합니다.

**컨볼루션 네트워크**를 활용하고 있기 때문에 (다른 유형들은 현재 지원되지 않음), 다음 설정을 통해 추론 전에 cuDNN autotuner를 활성화할 수 있습니다:

```python
import torch

torch.backends.cudnn.benchmark = True
```

### fp32 대신 tf32 사용하기  (Ampere 및 이후 CUDA 장치들에서)

Ampere 및 이후 CUDA 장치에서 행렬곱 및 컨볼루션은 TensorFloat32(TF32) 모드를 사용하여 더 빠르지만 약간 덜 정확할 수 있습니다.
기본적으로 PyTorch는 컨볼루션에 대해 TF32 모드를 활성화하지만 행렬 곱셈은 활성화하지 않습니다.
네트워크에 완전한 float32 정밀도가 필요한 경우가 아니면 행렬 곱셈에 대해서도 이 설정을 활성화하는 것이 좋습니다.
이는 일반적으로 무시할 수 있는 수치의 정확도 손실이 있지만, 계산 속도를 크게 높일 수 있습니다.
그것에 대해 [여기](https://huggingface.co/docs/transformers/v4.18.0/en/performance#tf32)서 더 읽을 수 있습니다.
추론하기 전에 다음을 추가하기만 하면 됩니다:

```python
import torch

torch.backends.cuda.matmul.allow_tf32 = True
```

## 반정밀도 가중치

더 많은 GPU 메모리를 절약하고 더 빠른 속도를 얻기 위해 모델 가중치를 반정밀도(half precision)로 직접 불러오고 실행할 수 있습니다.
여기에는 `fp16`이라는 브랜치에 저장된 float16 버전의 가중치를 불러오고, 그 때 `float16` 유형을 사용하도록 PyTorch에 지시하는 작업이 포함됩니다.

```Python
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",

    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
```

> [!WARNING]
> 어떤 파이프라인에서도 [`torch.autocast`](https://pytorch.org/docs/stable/amp.html#torch.autocast) 를 사용하는 것은 검은색 이미지를 생성할 수 있고, 순수한 float16 정밀도를 사용하는 것보다 항상 느리기 때문에 사용하지 않는 것이 좋습니다.

## 추가 메모리 절약을 위한 슬라이스 어텐션

추가 메모리 절약을 위해, 한 번에 모두 계산하는 대신 단계적으로 계산을 수행하는 슬라이스 버전의 어텐션(attention)을 사용할 수 있습니다.

> [!TIP]
> Attention slicing은 모델이 하나 이상의 어텐션 헤드를 사용하는 한, 배치 크기가 1인 경우에도 유용합니다.
>   하나 이상의 어텐션 헤드가 있는 경우 *QK^T* 어텐션 매트릭스는 상당한 양의 메모리를 절약할 수 있는 각 헤드에 대해 순차적으로 계산될 수 있습니다.

각 헤드에 대해 순차적으로 어텐션 계산을 수행하려면, 다음과 같이 추론 전에 파이프라인에서 [`~StableDiffusionPipeline.enable_attention_slicing`]를 호출하면 됩니다:

```Python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",

    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
pipe.enable_attention_slicing()
image = pipe(prompt).images[0]
```

추론 시간이 약 10% 느려지는 약간의 성능 저하가 있지만 이 방법을 사용하면 3.2GB 정도의 작은 VRAM으로도 Stable Diffusion을 사용할 수 있습니다!


## 더 큰 배치를 위한 sliced VAE 디코드

제한된 VRAM에서 대규모 이미지 배치를 디코딩하거나 32개 이상의 이미지가 포함된 배치를 활성화하기 위해, 배치의 latent 이미지를 한 번에 하나씩 디코딩하는 슬라이스 VAE 디코드를 사용할 수 있습니다.

이를 [`~StableDiffusionPipeline.enable_attention_slicing`] 또는 [`~StableDiffusionPipeline.enable_xformers_memory_efficient_attention`]과 결합하여 메모리 사용을 추가로 최소화할 수 있습니다.

VAE 디코드를 한 번에 하나씩 수행하려면 추론 전에 파이프라인에서 [`~StableDiffusionPipeline.enable_vae_slicing`]을 호출합니다. 예를 들어:

```Python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",

    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
pipe.enable_vae_slicing()
images = pipe([prompt] * 32).images
```

다중 이미지 배치에서 VAE 디코드가 약간의 성능 향상이 이루어집니다. 단일 이미지 배치에서는 성능 영향은 없습니다.


<a name="sequential_offloading"></a>
## 메모리 절약을 위해 가속 기능을 사용하여 CPU로 오프로딩

추가 메모리 절약을 위해 가중치를 CPU로 오프로드하고 순방향 전달을 수행할 때만 GPU로 로드할 수 있습니다.

CPU 오프로딩을 수행하려면 [`~StableDiffusionPipeline.enable_sequential_cpu_offload`]를 호출하기만 하면 됩니다:

```Python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",

    torch_dtype=torch.float16,
)

prompt = "a photo of an astronaut riding a horse on mars"
pipe.enable_sequential_cpu_offload()
image = pipe(prompt).images[0]
```

그러면 메모리 소비를 3GB 미만으로 줄일 수 있습니다.

참고로 이 방법은 전체 모델이 아닌 서브모듈 수준에서 작동합니다. 이는 메모리 소비를 최소화하는 가장 좋은 방법이지만 프로세스의 반복적 특성으로 인해 추론 속도가 훨씬 느립니다. 파이프라인의 UNet 구성 요소는 여러 번 실행됩니다('num_inference_steps' 만큼). 매번 UNet의 서로 다른 서브모듈이 순차적으로 온로드된 다음 필요에 따라 오프로드되므로 메모리 이동 횟수가 많습니다.

> [!TIP]
> 또 다른 최적화 방법인 <a href="#model_offloading">모델 오프로딩</a>을 사용하는 것을 고려하십시오. 이는 훨씬 빠르지만 메모리 절약이 크지는 않습니다.

또한 ttention slicing과 연결해서 최소 메모리(< 2GB)로도 동작할 수 있습니다.


```Python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",

    torch_dtype=torch.float16,
)

prompt = "a photo of an astronaut riding a horse on mars"
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing(1)

image = pipe(prompt).images[0]
```

**참고**: 'enable_sequential_cpu_offload()'를 사용할 때, 미리 파이프라인을 CUDA로 이동하지 **않는** 것이 중요합니다.그렇지 않으면 메모리 소비의 이득이 최소화됩니다. 더 많은 정보를 위해 [이 이슈](https://github.com/huggingface/diffusers/issues/1934)를 보세요.

<a name="model_offloading"></a>
## 빠른 추론과 메모리 메모리 절약을 위한 모델 오프로딩

[순차적 CPU 오프로딩](#sequential_offloading)은 이전 섹션에서 설명한 것처럼 많은 메모리를 보존하지만 필요에 따라 서브모듈을 GPU로 이동하고 새 모듈이 실행될 때 즉시 CPU로 반환되기 때문에 추론 속도가 느려집니다.

전체 모델 오프로딩은 각 모델의 구성 요소인 _modules_을 처리하는 대신, 전체 모델을 GPU로 이동하는 대안입니다. 이로 인해 추론 시간에 미치는 영향은 미미하지만(파이프라인을 'cuda'로 이동하는 것과 비교하여) 여전히 약간의 메모리를 절약할 수 있습니다.

이 시나리오에서는 파이프라인의 주요 구성 요소 중 하나만(일반적으로 텍스트 인코더, unet 및 vae) GPU에 있고, 나머지는 CPU에서 대기할 것입니다.
여러 반복을 위해 실행되는 UNet과 같은 구성 요소는 더 이상 필요하지 않을 때까지 GPU에 남아 있습니다.

이 기능은 아래와 같이 파이프라인에서 `enable_model_cpu_offload()`를 호출하여 활성화할 수 있습니다.

```Python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)

prompt = "a photo of an astronaut riding a horse on mars"
pipe.enable_model_cpu_offload()
image = pipe(prompt).images[0]
```

이는 추가적인 메모리 절약을 위한 attention slicing과도 호환됩니다.

```Python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)

prompt = "a photo of an astronaut riding a horse on mars"
pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing(1)

image = pipe(prompt).images[0]
```

> [!TIP]
> 이 기능을 사용하려면 'accelerate' 버전 0.17.0 이상이 필요합니다.

## Channels Last 메모리 형식 사용하기

Channels Last 메모리 형식은 차원 순서를 보존하는 메모리에서 NCHW 텐서 배열을 대체하는 방법입니다.
Channels Last 텐서는 채널이 가장 조밀한 차원이 되는 방식으로 정렬됩니다(일명 픽셀당 이미지를 저장).
현재 모든 연산자 Channels Last 형식을 지원하는 것은 아니라 성능이 저하될 수 있으므로, 사용해보고 모델에 잘 작동하는지 확인하는 것이 좋습니다.


예를 들어 파이프라인의 UNet 모델이 channels Last 형식을 사용하도록 설정하려면 다음을 사용할 수 있습니다:

```python
print(pipe.unet.conv_out.state_dict()["weight"].stride())  # (2880, 9, 3, 1)
pipe.unet.to(memory_format=torch.channels_last)  # in-place 연산
# 2번째 차원에서 스트라이드 1을 가지는 (2880, 1, 960, 320)로, 연산이 작동함을 증명합니다.
print(pipe.unet.conv_out.state_dict()["weight"].stride())
```

## 추적(tracing)

추적은 모델을 통해 예제 입력 텐서를 통해 실행되는데, 해당 입력이 모델의 레이어를 통과할 때 호출되는 작업을 캡처하여 실행 파일 또는 'ScriptFunction'이 반환되도록 하고, 이는 just-in-time 컴파일로 최적화됩니다.

UNet 모델을 추적하기 위해 다음을 사용할 수 있습니다:

```python
import time
import torch
from diffusers import StableDiffusionPipeline
import functools

# torch 기울기 비활성화
torch.set_grad_enabled(False)

# 변수 설정
n_experiments = 2
unet_runs_per_experiment = 50


# 입력 불러오기
def generate_inputs():
    sample = torch.randn((2, 4, 64, 64), device="cuda", dtype=torch.float16)
    timestep = torch.rand(1, device="cuda", dtype=torch.float16) * 999
    encoder_hidden_states = torch.randn((2, 77, 768), device="cuda", dtype=torch.float16)
    return sample, timestep, encoder_hidden_states


pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")
unet = pipe.unet
unet.eval()
unet.to(memory_format=torch.channels_last)  # Channels Last 메모리 형식 사용
unet.forward = functools.partial(unet.forward, return_dict=False)  # return_dict=False을 기본값으로 설정

# 워밍업
for _ in range(3):
    with torch.inference_mode():
        inputs = generate_inputs()
        orig_output = unet(*inputs)

# 추적
print("tracing..")
unet_traced = torch.jit.trace(unet, inputs)
unet_traced.eval()
print("done tracing")


# 워밍업 및 그래프 최적화
for _ in range(5):
    with torch.inference_mode():
        inputs = generate_inputs()
        orig_output = unet_traced(*inputs)


# 벤치마킹
with torch.inference_mode():
    for _ in range(n_experiments):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(unet_runs_per_experiment):
            orig_output = unet_traced(*inputs)
        torch.cuda.synchronize()
        print(f"unet traced inference took {time.time() - start_time:.2f} seconds")
    for _ in range(n_experiments):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(unet_runs_per_experiment):
            orig_output = unet(*inputs)
        torch.cuda.synchronize()
        print(f"unet inference took {time.time() - start_time:.2f} seconds")

# 모델 저장
unet_traced.save("unet_traced.pt")
```

그 다음, 파이프라인의 `unet` 특성을 다음과 같이 추적된 모델로 바꿀 수 있습니다.

```python
from diffusers import StableDiffusionPipeline
import torch
from dataclasses import dataclass


@dataclass
class UNet2DConditionOutput:
    sample: torch.Tensor


pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

# jitted unet 사용
unet_traced = torch.jit.load("unet_traced.pt")


# pipe.unet 삭제
class TracedUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = pipe.unet.config.in_channels
        self.device = pipe.unet.device

    def forward(self, latent_model_input, t, encoder_hidden_states):
        sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)


pipe.unet = TracedUNet()

with torch.inference_mode():
    image = pipe([prompt] * 1, num_inference_steps=50).images[0]
```


## Memory-efficient attention

어텐션 블록의 대역폭을 최적화하는 최근 작업으로 GPU 메모리 사용량이 크게 향상되고 향상되었습니다.
@tridao의 가장 최근의 플래시 어텐션: [code](https://github.com/HazyResearch/flash-attention), [paper](https://huggingface.co/papers/2205.14135).

배치 크기 1(프롬프트 1개)의 512x512 크기로 추론을 실행할 때 몇 가지 Nvidia GPU에서 얻은 속도 향상은 다음과 같습니다:

| GPU              	| 기준 어텐션 FP16 	       | 메모리 효율적인 어텐션 FP16 	|
|------------------	|---------------------	|---------------------------------	|
| NVIDIA Tesla T4  	| 3.5it/s             	| 5.5it/s                         	|
| NVIDIA 3060 RTX  	| 4.6it/s             	| 7.8it/s                         	|
| NVIDIA A10G      	| 8.88it/s            	| 15.6it/s                        	|
| NVIDIA RTX A6000 	| 11.7it/s            	| 21.09it/s                       	|
| NVIDIA TITAN RTX  | 12.51it/s         	| 18.22it/s                       	|
| A100-SXM4-40GB    	| 18.6it/s            	| 29.it/s                        	|
| A100-SXM-80GB    	| 18.7it/s            	| 29.5it/s                        	|

이를 활용하려면 다음을 만족해야 합니다:
 - PyTorch > 1.12
 - Cuda 사용 가능
 - [xformers 라이브러리를 설치함](xformers)
```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()

with torch.inference_mode():
    sample = pipe("a small cat")

# 선택: 이를 비활성화 하기 위해 다음을 사용할 수 있습니다.
# pipe.disable_xformers_memory_efficient_attention()
```
