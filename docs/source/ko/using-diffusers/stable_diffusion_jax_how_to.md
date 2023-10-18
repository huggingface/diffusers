<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# JAX / Flax에서의 🧨 Stable Diffusion!

[[open-in-colab]]

🤗 Hugging Face [Diffusers] (https://github.com/huggingface/diffusers) 는 버전 0.5.1부터 Flax를 지원합니다! 이를 통해 Colab, Kaggle, Google Cloud Platform에서 사용할 수 있는 것처럼 Google TPU에서 초고속 추론이 가능합니다.

이 노트북은 JAX / Flax를 사용해 추론을 실행하는 방법을 보여줍니다. Stable Diffusion의 작동 방식에 대한 자세한 내용을 원하거나 GPU에서 실행하려면 이 [노트북] ](https://huggingface.co/docs/diffusers/stable_diffusion)을 참조하세요.

먼저, TPU 백엔드를 사용하고 있는지 확인합니다. Colab에서 이 노트북을 실행하는 경우, 메뉴에서 런타임을 선택한 다음 "런타임 유형 변경" 옵션을 선택한 다음 하드웨어 가속기 설정에서 TPU를 선택합니다.

JAX는 TPU 전용은 아니지만 각 TPU 서버에는 8개의 TPU 가속기가 병렬로 작동하기 때문에 해당 하드웨어에서 더 빛을 발한다는 점은 알아두세요.


## Setup

먼저 diffusers가 설치되어 있는지 확인합니다.

```bash
!pip install jax==0.3.25 jaxlib==0.3.25 flax transformers ftfy
!pip install diffusers
```

```python
import jax.tools.colab_tpu

jax.tools.colab_tpu.setup_tpu()
import jax
```

```python
num_devices = jax.device_count()
device_type = jax.devices()[0].device_kind

print(f"Found {num_devices} JAX devices of type {device_type}.")
assert (
    "TPU" in device_type
), "Available device is not a TPU, please select TPU from Edit > Notebook settings > Hardware accelerator"
```

```python out
Found 8 JAX devices of type Cloud TPU.
```

그런 다음 모든 dependencies를 가져옵니다.

```python
import numpy as np
import jax
import jax.numpy as jnp

from pathlib import Path
from jax import pmap
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from PIL import Image

from huggingface_hub import notebook_login
from diffusers import FlaxStableDiffusionPipeline
```

## 모델 불러오기

TPU 장치는 효율적인 half-float 유형인 bfloat16을 지원합니다. 테스트에는 이 유형을 사용하지만 대신 float32를 사용하여 전체 정밀도(full precision)를 사용할 수도 있습니다.

```python
dtype = jnp.bfloat16
```

Flax는 함수형 프레임워크이므로 모델은 무상태(stateless)형이며 매개변수는 모델 외부에 저장됩니다. 사전학습된 Flax 파이프라인을 불러오면 파이프라인 자체와 모델 가중치(또는 매개변수)가 모두 반환됩니다. 저희는 bf16 버전의 가중치를 사용하고 있으므로 유형 경고가 표시되지만 무시해도 됩니다.

```python
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="bf16",
    dtype=dtype,
)
```

## 추론

TPU에는 일반적으로 8개의 디바이스가 병렬로 작동하므로 보유한 디바이스 수만큼 프롬프트를 복제합니다. 그런 다음 각각 하나의 이미지 생성을 담당하는 8개의 디바이스에서 한 번에 추론을 수행합니다. 따라서 하나의 칩이 하나의 이미지를 생성하는 데 걸리는 시간과 동일한 시간에 8개의 이미지를 얻을 수 있습니다.

프롬프트를 복제하고 나면 파이프라인의 `prepare_inputs` 함수를 호출하여 토큰화된 텍스트 ID를 얻습니다. 토큰화된 텍스트의 길이는 기본 CLIP 텍스트 모델의 구성에 따라 77토큰으로 설정됩니다.

```python
prompt = "A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of field, close up, split lighting, cinematic"
prompt = [prompt] * jax.device_count()
prompt_ids = pipeline.prepare_inputs(prompt)
prompt_ids.shape
```

```python out
(8, 77)
```

### 복사(Replication) 및 정렬화

모델 매개변수와 입력값은 우리가 보유한 8개의 병렬 장치에 복사(Replication)되어야 합니다. 매개변수 딕셔너리는 `flax.jax_utils.replicate`(딕셔너리를 순회하며 가중치의 모양을 변경하여 8번 반복하는 함수)를 사용하여 복사됩니다. 배열은 `shard`를 사용하여 복제됩니다.

```python
p_params = replicate(params)
```

```python
prompt_ids = shard(prompt_ids)
prompt_ids.shape
```

```python out
(8, 1, 77)
```

이 shape은 8개의 디바이스 각각이 shape `(1, 77)`의 jnp 배열을 입력값으로 받는다는 의미입니다. 즉 1은 디바이스당 batch(배치) 크기입니다. 메모리가 충분한 TPU에서는 한 번에 여러 이미지(칩당)를 생성하려는 경우 1보다 클 수 있습니다.

이미지를 생성할 준비가 거의 완료되었습니다! 이제 생성 함수에 전달할 난수 생성기만 만들면 됩니다. 이것은 난수를 다루는 모든 함수에 난수 생성기가 있어야 한다는, 난수에 대해 매우 진지하고 독단적인 Flax의 표준 절차입니다. 이렇게 하면 여러 분산된 기기에서 훈련할 때에도 재현성이 보장됩니다.

아래 헬퍼 함수는 시드를 사용하여 난수 생성기를 초기화합니다. 동일한 시드를 사용하는 한 정확히 동일한 결과를 얻을 수 있습니다. 나중에 노트북에서 결과를 탐색할 때엔 다른 시드를 자유롭게 사용하세요.

```python
def create_key(seed=0):
    return jax.random.PRNGKey(seed)
```

rng를 얻은 다음 8번 '분할'하여 각 디바이스가 다른 제너레이터를 수신하도록 합니다. 따라서 각 디바이스마다 다른 이미지가 생성되며 전체 프로세스를 재현할 수 있습니다.

```python
rng = create_key(0)
rng = jax.random.split(rng, jax.device_count())
```

JAX 코드는 매우 빠르게 실행되는 효율적인 표현으로 컴파일할 수 있습니다. 하지만 후속 호출에서 모든 입력이 동일한 모양을 갖도록 해야 하며, 그렇지 않으면 JAX가 코드를 다시 컴파일해야 하므로 최적화된 속도를 활용할 수 없습니다.

`jit = True`를 인수로 전달하면 Flax 파이프라인이 코드를 컴파일할 수 있습니다. 또한 모델이 사용 가능한 8개의 디바이스에서 병렬로 실행되도록 보장합니다.

다음 셀을 처음 실행하면 컴파일하는 데 시간이 오래 걸리지만 이후 호출(입력이 다른 경우에도)은 훨씬 빨라집니다. 예를 들어, 테스트했을 때 TPU v2-8에서 컴파일하는 데 1분 이상 걸리지만 이후 추론 실행에는 약 7초가 걸립니다.

```
%%time
images = pipeline(prompt_ids, p_params, rng, jit=True)[0]
```

```python out
CPU times: user 56.2 s, sys: 42.5 s, total: 1min 38s
Wall time: 1min 29s
```

반환된 배열의 shape은 `(8, 1, 512, 512, 3)`입니다. 이를 재구성하여 두 번째 차원을 제거하고 512 × 512 × 3의 이미지 8개를 얻은 다음 PIL로 변환합니다.

```python
images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
images = pipeline.numpy_to_pil(images)
```

### 시각화

이미지를 그리드에 표시하는 도우미 함수를 만들어 보겠습니다.

```python
def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
```

```python
image_grid(images, 2, 4)
```

![img](https://huggingface.co/datasets/YiYiXu/test-doc-assets/resolve/main/stable_diffusion_jax_how_to_cell_38_output_0.jpeg)


## 다른 프롬프트 사용

모든 디바이스에서 동일한 프롬프트를 복제할 필요는 없습니다. 프롬프트 2개를 각각 4번씩 생성하거나 한 번에 8개의 서로 다른 프롬프트를 생성하는 등 원하는 것은 무엇이든 할 수 있습니다. 한번 해보세요!

먼저 입력 준비 코드를 편리한 함수로 리팩터링하겠습니다:

```python
prompts = [
    "Labrador in the style of Hokusai",
    "Painting of a squirrel skating in New York",
    "HAL-9000 in the style of Van Gogh",
    "Times Square under water, with fish and a dolphin swimming around",
    "Ancient Roman fresco showing a man working on his laptop",
    "Close-up photograph of young black woman against urban background, high quality, bokeh",
    "Armchair in the shape of an avocado",
    "Clown astronaut in space, with Earth in the background",
]
```

```python
prompt_ids = pipeline.prepare_inputs(prompts)
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, p_params, rng, jit=True).images
images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
images = pipeline.numpy_to_pil(images)

image_grid(images, 2, 4)
```

![img](https://huggingface.co/datasets/YiYiXu/test-doc-assets/resolve/main/stable_diffusion_jax_how_to_cell_43_output_0.jpeg)


## 병렬화(parallelization)는 어떻게 작동하는가?

앞서 `diffusers` Flax 파이프라인이 모델을 자동으로 컴파일하고 사용 가능한 모든 기기에서 병렬로 실행한다고 말씀드렸습니다. 이제 그 프로세스를 간략하게 살펴보고 작동 방식을 보여드리겠습니다.

JAX 병렬화는 여러 가지 방법으로 수행할 수 있습니다. 가장 쉬운 방법은 jax.pmap 함수를 사용하여 단일 프로그램, 다중 데이터(SPMD) 병렬화를 달성하는 것입니다. 즉, 동일한 코드의 복사본을 각각 다른 데이터 입력에 대해 여러 개 실행하는 것입니다. 더 정교한 접근 방식도 가능하므로 관심이 있으시다면 [JAX 문서](https://jax.readthedocs.io/en/latest/index.html)와 [`pjit` 페이지](https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html?highlight=pjit)에서 이 주제를 살펴보시기 바랍니다!

`jax.pmap`은 두 가지 기능을 수행합니다:

- `jax.jit()`를 호출한 것처럼 코드를 컴파일(또는 `jit`)합니다. 이 작업은 `pmap`을 호출할 때가 아니라 pmapped 함수가 처음 호출될 때 수행됩니다.
- 컴파일된 코드가 사용 가능한 모든 기기에서 병렬로 실행되도록 합니다.

작동 방식을 보여드리기 위해 이미지 생성을 실행하는 비공개 메서드인 파이프라인의 `_generate` 메서드를 `pmap`합니다. 이 메서드는 향후 `Diffusers` 릴리스에서 이름이 변경되거나 제거될 수 있다는 점에 유의하세요.

```python
p_generate = pmap(pipeline._generate)
```

`pmap`을 사용한 후 준비된 함수 `p_generate`는 개념적으로 다음을 수행합니다:
* 각 장치에서 기본 함수 `pipeline._generate`의 복사본을 호출합니다.
* 각 장치에 입력 인수의 다른 부분을 보냅니다. 이것이 바로 샤딩이 사용되는 이유입니다. 이 경우 `prompt_ids`의 shape은 `(8, 1, 77, 768)`입니다. 이 배열은 8개로 분할되고 `_generate`의 각 복사본은 `(1, 77, 768)`의 shape을 가진 입력을 받게 됩니다.

병렬로 호출된다는 사실을 완전히 무시하고 `_generate`를 코딩할 수 있습니다. batch(배치) 크기(이 예제에서는 `1`)와 코드에 적합한 차원만 신경 쓰면 되며, 병렬로 작동하기 위해 아무것도 변경할 필요가 없습니다.

파이프라인 호출을 사용할 때와 마찬가지로, 다음 셀을 처음 실행할 때는 시간이 걸리지만 그 이후에는 훨씬 빨라집니다.

```
%%time
images = p_generate(prompt_ids, p_params, rng)
images = images.block_until_ready()
images.shape
```

```python out
CPU times: user 1min 15s, sys: 18.2 s, total: 1min 34s
Wall time: 1min 15s
```

```python
images.shape
```

```python out
(8, 1, 512, 512, 3)
```

JAX는 비동기 디스패치를 사용하고 가능한 한 빨리 제어권을 Python 루프에 반환하기 때문에 추론 시간을 정확하게 측정하기 위해 `block_until_ready()`를 사용합니다. 아직 구체화되지 않은 계산 결과를 사용하려는 경우 자동으로 차단이 수행되므로 코드에서 이 함수를 사용할 필요가 없습니다.