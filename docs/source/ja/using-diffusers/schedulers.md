<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# スケジューラ

[[open-in-colab]]

拡散パイプラインは、拡散モデルとスケジューラの集合体であり、違いに部分的に独立しています。
つまり、パイプラインの一部を切り替えて、自分のユースケースに合わせてパイプラインをカスタマイズすることができます。
その最たる例として、[スケジューラ](../api/schedulers/overview)が挙げられます。

拡散モデルは通常、ノイズからよりノイズの少ないサンプルへのシンプルなフォワードパスを定義するだけです。
それに対して、スケジューラはノイズ除去過程全体を定義します:
- ノイズ除去を何ステップ行うのか?
- 確率論的 (Stochastic) か決定論的 (Deterministic) か?
- どんなアルゴリズムでノイズ除去後のサンプルを見つけるのか?

これらは非常に複雑で、しばしば**ノイズ除去速度**と**ノイズ除去品質**の間でトレードオフが発生します。
ある拡散パイプラインに対して、どのスケジューラが最適かを定量的に測ることは非常に困難なため、シンプルにどれが最適かを試してみることが推奨されます。

以下では、それらの試みを 🧨 Diffusers ライブラリでどのように実現するかを紹介します。

## パイプラインの読み込み

はじめに、[`DiffusionPipeline`] で [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) モデルを読み込みます。

```python
from huggingface_hub import login
from diffusers import DiffusionPipeline
import torch

login()

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
```

次に、モデルを GPU に移動させます:

```python
pipeline.to("cuda")
```

## スケジューラへアクセスする

スケジューラは、常にパイプラインのコンポーネントの1つであり、通常 `"scheduler"` と呼ばれる。
そのため、`"scheduler"` プロパティによってアクセスできます:

```python
pipeline.scheduler
```

**Output**:
```
PNDMScheduler {
  "_class_name": "PNDMScheduler",
  "_diffusers_version": "0.21.4",
  "beta_end": 0.012,
  "beta_schedule": "scaled_linear",
  "beta_start": 0.00085,
  "clip_sample": false,
  "num_train_timesteps": 1000,
  "set_alpha_to_one": false,
  "skip_prk_steps": true,
  "steps_offset": 1,
  "timestep_spacing": "leading",
  "trained_betas": null
}
```

このスケジューラは [`PNDMScheduler`] であることがわかります。
それでは、このスケジューラを他のスケジューラと比較してみましょう。
まず、全ての異なるスケジューラをテストするプロンプトを定義します:

```python
prompt = "A photograph of an astronaut riding a horse on Mars, high resolution, high definition."
```

次に、パイプラインを実行するだけではなく、類似した画像が生成できるように、ランダムなシードからジェネレータを作成します:

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


## スケジューラを変更する

ここでは、パイプラインのスケジューラを変更することがいかに簡単であるかを示します。
全てのスケジューラは、[`~SchedulerMixin.compatibles`] プロパティを持っており、互換性のある全てのスケジューラを定義しています。
Stable Diffusion パイプラインで利用可能な互換性のある全てのスケジューラは以下のようになっています。

```python
pipeline.scheduler.compatibles
```

**Output**:
```
[diffusers.utils.dummy_torch_and_torchsde_objects.DPMSolverSDEScheduler,
 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler,
 diffusers.schedulers.scheduling_lms_discrete.LMSDiscreteScheduler,
 diffusers.schedulers.scheduling_ddim.DDIMScheduler,
 diffusers.schedulers.scheduling_ddpm.DDPMScheduler,
 diffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler,
 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler,
 diffusers.schedulers.scheduling_deis_multistep.DEISMultistepScheduler,
 diffusers.schedulers.scheduling_pndm.PNDMScheduler,
 diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler,
 diffusers.schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler,
 diffusers.schedulers.scheduling_k_dpm_2_discrete.KDPM2DiscreteScheduler,
 diffusers.schedulers.scheduling_dpmsolver_singlestep.DPMSolverSinglestepScheduler,
 diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete.KDPM2AncestralDiscreteScheduler]
```

いいですね、多くのスケジューラがあるようです。
それぞれのクラス定義を見ていきましょう。

- [`EulerDiscreteScheduler`],
- [`LMSDiscreteScheduler`],
- [`DDIMScheduler`],
- [`DDPMScheduler`],
- [`HeunDiscreteScheduler`],
- [`DPMSolverMultistepScheduler`],
- [`DEISMultistepScheduler`],
- [`PNDMScheduler`],
- [`EulerAncestralDiscreteScheduler`],
- [`UniPCMultistepScheduler`],
- [`KDPM2DiscreteScheduler`],
- [`DPMSolverSinglestepScheduler`],
- [`KDPM2AncestralDiscreteScheduler`].

ここで、入力プロンプトを他のすべてのスケジューラと比較します。
パイプラインのスケジューラを変更するには、便利な [`~ConfigMixin.config`] プロパティと、[`~ConfigMixin.from_config`] 関数を組み合わせて使うことができます。

```python
pipeline.scheduler.config
```

スケジューラの設定を辞書として返されます。


**Output**:
```py
FrozenDict([('num_train_timesteps', 1000),
            ('beta_start', 0.00085),
            ('beta_end', 0.012),
            ('beta_schedule', 'scaled_linear'),
            ('trained_betas', None),
            ('skip_prk_steps', True),
            ('set_alpha_to_one', False),
            ('prediction_type', 'epsilon'),
            ('timestep_spacing', 'leading'),
            ('steps_offset', 1),
            ('_use_default_values', ['timestep_spacing', 'prediction_type']),
            ('_class_name', 'PNDMScheduler'),
            ('_diffusers_version', '0.21.4'),
            ('clip_sample', False)])
```

この設定を使うと、パイプラインと互換性のある別のクラスのスケジューラをインスタンス化することができます。
ここでは、スケジューラを [`DDIMScheduler`] に変更してみましょう。


```python
from diffusers import DDIMScheduler

pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
```

いいですね。もう一度パイプラインを実行してみることで、生成品質の違いを比較することができます。

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

もし、あなたが JAX や Flax を使っているのであれば、代わりに [このセクション](#changing-the-scheduler-in-flax) をご確認ください。

## スケジューラの比較

ここまで、[`PNDMScheduler`] と [`DDIMScheduler`] の2つスケジューラで Stable Diffusion パイプラインを実行してみました。
より少ないステップ数で実行できる、優れたスケジューラが数多くリリースされているので、ここで比較してみましょう:

[`LMSDiscreteScheduler`] は、通常よい良い結果を示します:

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


[`EulerDiscreteScheduler`] と [`EulerAncestralDiscreteScheduler`] は、わずか30ステップで高品質な結果を生成することができます。

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


そして:

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


[`DPMSolverMultistepScheduler`] は、わずか20ステップで実行でき、速度と品質のトレードオフの合理的なバランスを実現しています。

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

ご覧のように、ほどんどの画像は非常によく似ており、間違いなく同等の品質であると言えます。
どのスケジューラを選択するかは、特定のユースケースに依存することが多く、
常に複数のスケジューラで実行してみて、結果を比較することが良いアプローチです。

## Flax でスケジューラを変更する

あなたが JAX や Flax を利用しているなら、デフォルトのパイプラインのスケジューラを変更することもできます。
これは、Flax Stable Diffusion パイプラインと、超高速な [DPM-Solver++ scheduler](../api/schedulers/multistep_dpm_solver) を使った推論の例です:

```Python
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard

from diffusers import FlaxStableDiffusionPipeline, FlaxDPMSolverMultistepScheduler

model_id = "runwayml/stable-diffusion-v1-5"
scheduler, scheduler_state = FlaxDPMSolverMultistepScheduler.from_pretrained(
    model_id,
    subfolder="scheduler"
)
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    revision="bf16",
    dtype=jax.numpy.bfloat16,
)
params["scheduler"] = scheduler_state

# 並列デバイスごとに1つの画像を生成します（TPUv2-8またはTPUv3-8では8つ）。
prompt = "a photo of an astronaut riding a horse on mars"
num_samples = jax.device_count()
prompt_ids = pipeline.prepare_inputs([prompt] * num_samples)

prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 25

# シャード入力と RNG
params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
```

<Tip warning={true}>

以下のFlaxスケジューラは、Flax Stable Diffusion Pipelineと _まだ互換性がありません_ :

- `FlaxLMSDiscreteScheduler`
- `FlaxDDPMScheduler`

</Tip>
