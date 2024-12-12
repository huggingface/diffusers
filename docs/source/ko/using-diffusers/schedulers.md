<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 스케줄러

diffusion 파이프라인은 diffusion 모델, 스케줄러 등의 컴포넌트들로 구성됩니다. 그리고 파이프라인 안의 일부 컴포넌트를 다른 컴포넌트로 교체하는 식의 커스터마이징 역시 가능합니다.  이와 같은 컴포넌트 커스터마이징의 가장 대표적인 예시가 바로 [스케줄러](../api/schedulers/overview.md)를 교체하는 것입니다.



스케쥴러는 다음과 같이 diffusion 시스템의 전반적인 디노이징 프로세스를 정의합니다.

- 디노이징 스텝을 얼마나 가져가야 할까?
- 확률적으로(stochastic) 혹은 확정적으로(deterministic)?
- 디노이징 된 샘플을 찾아내기 위해 어떤 알고리즘을 사용해야 할까?

이러한 프로세스는 다소 난해하고, 디노이징 속도와 디노이징 퀄리티 사이의 트레이드 오프를 정의해야 하는 문제가 될 수 있습니다. 주어진 파이프라인에 어떤 스케줄러가 가장 적합한지를 정량적으로 판단하는 것은 매우 어려운 일입니다. 이로 인해 일단 해당 스케줄러를 직접 사용하여, 생성되는 이미지를 직접 눈으로 보며, 정성적으로 성능을 판단해보는 것이 추천되곤 합니다.





## 파이프라인 불러오기

먼저 스테이블 diffusion 파이프라인을 불러오도록 해보겠습니다. 물론 스테이블 diffusion을 사용하기 위해서는, 허깅페이스 허브에 등록된 사용자여야 하며, 관련 [라이센스](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)에 동의해야 한다는 점을 잊지 말아주세요.

*역자 주: 다만, 현재 신규로 생성한 허깅페이스 계정에 대해서는 라이센스 동의를 요구하지 않는 것으로 보입니다!*

```python
from huggingface_hub import login
from diffusers import DiffusionPipeline
import torch

# first we need to login with our access token
login()

# Now we can download the pipeline
pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
```

다음으로, GPU로 이동합니다.

```python
pipeline.to("cuda")
```





## 스케줄러 액세스

스케줄러는 언제나 파이프라인의 컴포넌트로서 존재하며, 일반적으로 파이프라인 인스턴스 내에 `scheduler`라는 이름의 속성(property)으로 정의되어 있습니다.

```python
pipeline.scheduler
```

**Output**:

```
PNDMScheduler {
  "_class_name": "PNDMScheduler",
  "_diffusers_version": "0.8.0.dev0",
  "beta_end": 0.012,
  "beta_schedule": "scaled_linear",
  "beta_start": 0.00085,
  "clip_sample": false,
  "num_train_timesteps": 1000,
  "set_alpha_to_one": false,
  "skip_prk_steps": true,
  "steps_offset": 1,
  "trained_betas": null
}
```

출력 결과를 통해, 우리는 해당 스케줄러가 [`PNDMScheduler`]의 인스턴스라는 것을 알 수 있습니다. 이제 [`PNDMScheduler`]와 다른 스케줄러들의 성능을 비교해보도록 하겠습니다. 먼저 테스트에 사용할 프롬프트를 다음과 같이 정의해보도록 하겠습니다.

```python
prompt = "A photograph of an astronaut riding a horse on Mars, high resolution, high definition."
```

다음으로 유사한 이미지 생성을 보장하기 위해서, 다음과 같이 랜덤시드를 고정해주도록 하겠습니다.

```python
generator = torch.Generator(device="cuda").manual_seed(8)
image = pipeline(prompt, generator=generator).images[0]
image
```

<p align="center">
    <br>
    <img src="https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_pndm.png" width="400"/>
    <br>
</p>




## 스케줄러 교체하기

다음으로 파이프라인의 스케줄러를 다른 스케줄러로 교체하는 방법에 대해 알아보겠습니다. 모든 스케줄러는 [`SchedulerMixin.compatibles`]라는 속성(property)을 갖고 있습니다. 해당 속성은 **호환 가능한** 스케줄러들에 대한 정보를 담고 있습니다.

```python
pipeline.scheduler.compatibles
```

**Output**:

```
[diffusers.schedulers.scheduling_lms_discrete.LMSDiscreteScheduler,
 diffusers.schedulers.scheduling_ddim.DDIMScheduler,
 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler,
 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler,
 diffusers.schedulers.scheduling_pndm.PNDMScheduler,
 diffusers.schedulers.scheduling_ddpm.DDPMScheduler,
 diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler]
```

호환되는 스케줄러들을 살펴보면 아래와 같습니다.

- [`LMSDiscreteScheduler`],
- [`DDIMScheduler`],
- [`DPMSolverMultistepScheduler`],
- [`EulerDiscreteScheduler`],
- [`PNDMScheduler`],
- [`DDPMScheduler`],
- [`EulerAncestralDiscreteScheduler`].

앞서 정의했던 프롬프트를 사용해서 각각의 스케줄러들을 비교해보도록 하겠습니다.

먼저 파이프라인 안의 스케줄러를 바꾸기 위해 [`ConfigMixin.config`] 속성과 [`ConfigMixin.from_config`] 메서드를 활용해보려고 합니다.



```python
pipeline.scheduler.config
```

**Output**:

```
FrozenDict([('num_train_timesteps', 1000),
            ('beta_start', 0.00085),
            ('beta_end', 0.012),
            ('beta_schedule', 'scaled_linear'),
            ('trained_betas', None),
            ('skip_prk_steps', True),
            ('set_alpha_to_one', False),
            ('steps_offset', 1),
            ('_class_name', 'PNDMScheduler'),
            ('_diffusers_version', '0.8.0.dev0'),
            ('clip_sample', False)])
```

기존 스케줄러의 config를 호환 가능한 다른 스케줄러에 이식하는 것 역시 가능합니다.

다음 예시는 기존 스케줄러(`pipeline.scheduler`)를 다른 종류의 스케줄러(`DDIMScheduler`)로 바꾸는 코드입니다. 기존 스케줄러가 갖고 있던 config를 `.from_config` 메서드의 인자로 전달하는 것을 확인할 수 있습니다.

```python
from diffusers import DDIMScheduler

pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
```



이제 파이프라인을 실행해서 두 스케줄러 사이의 생성된 이미지의 퀄리티를 비교해봅시다.

```python
generator = torch.Generator(device="cuda").manual_seed(8)
image = pipeline(prompt, generator=generator).images[0]
image
```

<p align="center">
    <br>
    <img src="https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_ddim.png" width="400"/>
    <br>
</p>




## 스케줄러들 비교해보기

지금까지는 [`PNDMScheduler`]와 [`DDIMScheduler`] 스케줄러를 실행해보았습니다. 아직 비교해볼 스케줄러들이 더 많이 남아있으니 계속 비교해보도록 하겠습니다.



[`LMSDiscreteScheduler`]을 일반적으로 더 좋은 결과를 보여줍니다.

```python
from diffusers import LMSDiscreteScheduler

pipeline.scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)

generator = torch.Generator(device="cuda").manual_seed(8)
image = pipeline(prompt, generator=generator).images[0]
image
```

<p align="center">
    <br>
    <img src="https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_lms.png" width="400"/>
    <br>
</p>


[`EulerDiscreteScheduler`]와 [`EulerAncestralDiscreteScheduler`] 고작 30번의 inference step만으로도 높은 퀄리티의 이미지를 생성하는 것을 알 수 있습니다.

```python
from diffusers import EulerDiscreteScheduler

pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

generator = torch.Generator(device="cuda").manual_seed(8)
image = pipeline(prompt, generator=generator, num_inference_steps=30).images[0]
image
```

<p align="center">
    <br>
    <img src="https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_euler_discrete.png" width="400"/>
    <br>
</p>


```python
from diffusers import EulerAncestralDiscreteScheduler

pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

generator = torch.Generator(device="cuda").manual_seed(8)
image = pipeline(prompt, generator=generator, num_inference_steps=30).images[0]
image
```

<p align="center">
    <br>
    <img src="https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_euler_ancestral.png" width="400"/>
    <br>
</p>


지금 이 문서를 작성하는 현시점 기준에선, [`DPMSolverMultistepScheduler`]가 시간 대비 가장 좋은 품질의 이미지를 생성하는 것 같습니다. 20번 정도의 스텝만으로도 실행될 수 있습니다.



```python
from diffusers import DPMSolverMultistepScheduler

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

generator = torch.Generator(device="cuda").manual_seed(8)
image = pipeline(prompt, generator=generator, num_inference_steps=20).images[0]
image
```

<p align="center">
    <br>
    <img src="https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_dpm.png" width="400"/>
    <br>
</p>


보시다시피 생성된 이미지들은 매우 비슷하고, 비슷한 퀄리티를 보이는 것 같습니다. 실제로 어떤 스케줄러를 선택할 것인가는 종종 특정 이용 사례에 기반해서 결정되곤 합니다. 결국 여러 종류의 스케줄러를 직접 실행시켜보고 눈으로 직접 비교해서 판단하는 게 좋은 선택일 것 같습니다.



## Flax에서 스케줄러 교체하기

JAX/Flax 사용자인 경우 기본 파이프라인 스케줄러를 변경할 수도 있습니다. 다음은 Flax Stable Diffusion 파이프라인과 초고속 [DDPM-Solver++ 스케줄러를](../api/schedulers/multistep_dpm_solver) 사용하여 추론을 실행하는 방법에 대한 예시입니다 .

```Python
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard

from diffusers import FlaxStableDiffusionPipeline, FlaxDPMSolverMultistepScheduler

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
scheduler, scheduler_state = FlaxDPMSolverMultistepScheduler.from_pretrained(
    model_id,
    subfolder="scheduler"
)
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    variant="bf16",
    dtype=jax.numpy.bfloat16,
)
params["scheduler"] = scheduler_state

# Generate 1 image per parallel device (8 on TPUv2-8 or TPUv3-8)
prompt = "a photo of an astronaut riding a horse on mars"
num_samples = jax.device_count()
prompt_ids = pipeline.prepare_inputs([prompt] * num_samples)

prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 25

# shard inputs and rng
params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
```

<Tip warning={true}>

다음 Flax 스케줄러는 *아직* Flax Stable Diffusion 파이프라인과 호환되지 않습니다.

- `FlaxLMSDiscreteScheduler`
- `FlaxDDPMScheduler`

</Tip>

