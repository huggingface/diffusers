# تحميل المخططات الزمنية والنماذج 

تمثل خطوط أنابيب الانتشار مجموعة من المخططات الزمنية والنماذج القابلة للتبديل التي يمكن مزجها ومطابقتها لتخصيص خط أنابيب لحالة استخدام معينة. ويحتوي المخطط الزمني على عملية إزالة التشويش بأكملها، مثل عدد خطوات إزالة التشويش والخوارزمية للعثور على العينة الخالية من التشويش. لا يحتوي المخطط الزمني على معلمات أو تدريب، لذا فهو لا يستهلك الكثير من الذاكرة. عادة ما يهتم النموذج بمرور الإشارة الأمامية فقط من الإدخال المشوش إلى عينة أقل تشويشًا.

سيوضح لك هذا الدليل كيفية تحميل المخططات الزمنية والنماذج لتخصيص خط أنابيب. ستستخدم نقطة المراقبة [runwayml/stable-diffusion-v1-5](https://hf.co/runwayml/stable-diffusion-v1-5) في جميع أنحاء هذا الدليل، لذا دعنا نقوم بتحميلها أولاً.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
```

يمكنك معرفة المخطط الزمني الذي يستخدمه خط الأنابيب هذا باستخدام سمة `pipeline.scheduler`.

```py
pipeline.scheduler
PNDMScheduler {
"_class_name": "PNDMScheduler"،
"_diffusers_version": "0.21.4"،
"beta_end": 0.012،
"beta_schedule": "scaled_linear"،
"beta_start": 0.00085،
"clip_sample": false،
"num_train_timesteps": 1000،
"set_alpha_to_one": false،
"skip_prk_steps": true،
"steps_offset": 1،
"timestep_spacing": "leading"،
"trained_betas": null
}
```

## تحميل مخطط زمني

يتم تحديد المخططات الزمنية بواسطة ملف تكوين يمكن استخدامه بواسطة مجموعة متنوعة من المخططات الزمنية. قم بتحميل مخطط زمني باستخدام طريقة [`SchedulerMixin.from_pretrained`]، وحدد معلمة `subfolder` لتحميل ملف التكوين في المجلد الفرعي الصحيح لمستودع خط الأنابيب.

على سبيل المثال، لتحميل [`DDIMScheduler`]:

```py
from diffusers import DDIMScheduler, DiffusionPipeline

ddim = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
```

بعد ذلك، يمكنك تمرير المخطط الزمني المحمل حديثًا إلى خط الأنابيب.

```python
pipeline = DiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5"، scheduler=ddim، torch_dtype=torch.float16، use_safetensors=True
).to("cuda")
```

## مقارنة المخططات الزمنية

تتمتع المخططات الزمنية بمزايا وعيوب فريدة، مما يجعل من الصعب إجراء مقارنة كمية لتحديد المخطط الزمني الذي يعمل بشكل أفضل لخط الأنابيب. عادة ما يتعين عليك إجراء مفاضلة بين سرعة إزالة التشويش وجودة إزالة التشويش. نوصي بتجربة مخططات زمنية مختلفة للعثور على أفضل ما يناسب حالتك الاستخدامية. قم بالاتصال بسمة `pipeline.scheduler.compatibles` لمعرفة المخططات الزمنية المتوافقة مع خط الأنابيب.

دعونا نقارن [`LMSDiscreteScheduler`]، [`EulerDiscreteScheduler`]، [`EulerAncestralDiscreteScheduler`]، و [`DPMSolverMultistepScheduler`] على الفور والصورة التالية والبذرة.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5"، torch_dtype=torch.float16، use_safetensors=True
).to("cuda")

prompt = "صورة فوتوغرافية لرائد فضاء يركب حصانًا على المريخ، بدقة عالية، ووضوح عالٍ."
generator = torch.Generator(device="cuda").manual_seed(8)
```

لتغيير مخطط خط الأنابيب الزمني، استخدم طريقة [`~ConfigMixin.from_config`] لتحميل تكوين مخطط زمني مختلف في `pipeline.scheduler.config` إلى خط الأنابيب.

<hfoptions id="schedulers">
<hfoption id="LMSDiscreteScheduler">

عادةً ما يقوم [`LMSDiscreteScheduler`] بتوليد صور ذات جودة أعلى من المخطط الزمني الافتراضي.

```py
from diffusers import LMSDiscreteScheduler

pipeline.scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
image = pipeline(prompt, generator=generator).images[0]
image
```

</hfoption>

<hfoption id="EulerDiscreteScheduler">

يمكن لـ [`EulerDiscreteScheduler`] إنشاء صور ذات جودة أعلى في 30 خطوة فقط.

```py
from diffusers import EulerDiscreteScheduler

pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
image = pipeline(prompt, generator=generator).images[0]
image
```

</hfoption>

<hfoption id="EulerAncestralDiscreteScheduler">

يمكن لـ [`EulerAncestralDiscreteScheduler`] إنشاء صور ذات جودة أعلى في 30 خطوة فقط.

```py
from diffusers import EulerAncestralDiscreteScheduler

pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
image = pipeline(prompt, generator=generator).images[0]
image
```

</hfoption>

<hfoption id="DPMSolverMultistepScheduler">

يوفر [`DPMSolverMultistepScheduler`] توازنًا بين السرعة والجودة ويمكنه إنشاء صور ذات جودة أعلى في 20 خطوة فقط.

```py
from diffusers import DPMSolverMultistepScheduler

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
image = pipeline(prompt, generator=generator).images[0]
image
```

</hfoption>
</hfoptions>

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_lms.png" />
<figcaption class="mt-2 text-center text-sm text-gray-500">LMSDiscreteScheduler</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_euler_discrete.png" />
<figcaption class="mt-2 text-center text-sm text-gray-500">EulerDiscreteScheduler</figcaption>
</div>
</div>
<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_euler_ancestral.png" />
<figcaption class="mt-₂ text-center text-sm text-gray-500">EulerAncestralDiscreteScheduler</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_dpm.png" />
<figcaption class="mt-2 text-center text-sm text-gray-500">DPMSolverMultistepScheduler</figcaption>
</div>
</div>

تبدو معظم الصور متشابهة جدًا ومتقاربة في الجودة. مرة أخرى، غالبًا ما يتعلق الأمر بحالة الاستخدام المحددة الخاصة بك، لذا فإن أحد الأساليب الجيدة هو تشغيل العديد من المخططات الزمنية المختلفة ومقارنة النتائج.

### مخططات Flax الزمنية

لمقارنة مخططات Flax الزمنية، يلزم تحميل حالة المخطط الزمني في معلمات النموذج. على سبيل المثال، دعنا نغير المخطط الزمني الافتراضي في [`FlaxStableDiffusionPipeline`] لاستخدام [`FlaxDPMSolverMultistepScheduler`] فائق السرعة.

> [!تحذير]
> [`FlaxLMSDiscreteScheduler`] و [`FlaxDDPMScheduler`] غير متوافقين مع [`FlaxStableDiffusionPipeline`] بعد.

```py
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import FlaxStableDiffusionPipeline, FlaxDPMSolverMultistepScheduler

scheduler, scheduler_state = FlaxDPMSolverMultistepScheduler.from_pretrained(
"runwayml/stable-diffusion-v1-5"،
subfolder="scheduler"
)
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5"،
scheduler=scheduler،
revision="bf16"،
dtype=jax.numpy.bfloat16،
)
params["scheduler"] = scheduler_state
```

بعد ذلك، يمكنك الاستفادة من توافق Flax مع TPUs لإنشاء عدد من الصور بشكل متوازي. ستحتاج إلى عمل نسخة من معلمات النموذج لكل جهاز متاح، ثم تقسيم الإدخالات بينها لإنشاء العدد المرغوب من الصور.

```py
# إنشاء 1 صورة لكل جهاز متوازي (8 على TPUv2-8 أو TPUv3-8)
prompt = "صورة فوتوغرافية لرائد فضاء يركب حصانًا على المريخ، بدقة عالية، ووضوح عالٍ."
num_samples = jax.device_count()
prompt_ids = pipeline.prepare_inputs([prompt] * num_samples)

prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 25

# تقسيم الإدخالات و rng
params = replicate(params)
prng_seed = jax.random.split(prng_seed، jax.device_count())
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids، params، prng_seed، num_inference_steps، jit=True).images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
```

## النماذج

يتم تحميل النماذج من طريقة [`ModelMixin.from_pretrained`]، والتي تقوم بتنزيل وتخزين أحدث إصدار من أوزان النماذج وتكويناتها. إذا كانت أحدث الملفات متاحة في ذاكرة التخزين المؤقت المحلية، فإن [`~ModelMixin.from_pretrained`] يعيد استخدام الملفات الموجودة في الذاكرة المؤقتة بدلاً من إعادة تنزيلها.

يمكن تحميل النماذج من مجلد فرعي باستخدام حجة `subfolder`. على سبيل المثال، يتم تخزين أوزان النموذج لـ [runwayml/stable-diffusion-v1-5](https://hf.co/runwayml/stable-diffusion-v1-5) في المجلد الفرعي [unet](https://hf.co/runwayml/stable-diffusion-v1-5/tree/main/unet).

```python
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5"، subfolder="unet"، use_safetensors=True)
```

يمكن أيضًا تحميلها مباشرة من [مستودع](https://huggingface.co/google/ddpm-cifar10-32/tree/main).

```python
from diffusers import UNet2DModel

unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32"، use_safetensors=True)
```

لتحميل وحفظ متغيرات النموذج، حدد حجة `variant` في [`ModelMixin.from_pretrained`] و [`ModelMixin.save_pretrained`].

```python
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained(
"runwayml/stable-diffusion-v1-5"، subfolder="unet"، variant="non_ema"، use_safetensors=True
)
unet.save_pretrained("./local-unet"، variant="non_ema")
```