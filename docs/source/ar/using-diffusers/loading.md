
# تحميل الأنابيب (Pipelines)

تتكون أنظمة الانتشار من مكونات متعددة مثل النماذج المعلمية (parameterized models) والجداول الزمنية (schedulers) التي تتفاعل بطرق معقدة. ولهذا صممنا [`DiffusionPipeline`] لتبسيط التعقيد الكامل لنظام الانتشار في واجهة برمجة تطبيقات سهلة الاستخدام. وفي الوقت نفسه، يمكن تخصيص [`DiffusionPipeline`] بالكامل، بحيث يمكنك تعديل كل مكون لبناء نظام انتشار يناسب احتياجاتك.

سيوضح هذا الدليل كيفية التحميل:

- الأنابيب من Hub ومحليا
- مكونات مختلفة في الأنبوب
- أنابيب متعددة دون زيادة استخدام الذاكرة
- متغيرات نقاط التفتيش مثل أنواع النقاط العائمة المختلفة أو أوزان المتوسط غير الأسية (non-exponential mean averaged)

## تحميل خط أنابيب

> [!TIP]
> انتقل إلى قسم [شرح DiffusionPipeline](#diffusionpipeline-explained) إذا كنت مهتمًا بشرح حول كيفية عمل فئة [`DiffusionPipeline`].

هناك طريقتان لتحميل خط أنابيب لمهمة:

1. قم بتحميل فئة [`DiffusionPipeline`] العامة واسمح لها بالكشف التلقائي عن فئة خط الأنابيب الصحيحة من نقطة التفتيش.
2. قم بتحميل فئة خط أنابيب محددة لمهمة محددة.

<hfoptions id="pipelines">
<hfoption id="generic pipeline">

تمثل فئة [`DiffusionPipeline`] طريقة بسيطة وعامة لتحميل أحدث نموذج انتشار شائع من [Hub](https://huggingface.co/models?library=diffusers&sort=trending). يستخدم الطريقة [`~DiffusionPipeline.from_pretrained`] للكشف التلقائي عن فئة خط أنابيب المهمة الصحيحة من نقطة التفتيش، ويقوم بتنزيل وتخزين جميع ملفات التكوين والوزن المطلوبة، ويعيد خط أنابيب جاهزًا للاستدلال.

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
```

يمكن أيضًا استخدام نقطة التفتيش هذه لمهمة الصورة إلى الصورة. يمكن لفئة [`DiffusionPipeline`] التعامل مع أي مهمة طالما أنك توفر المدخلات المناسبة. على سبيل المثال، لمهمة الصورة إلى الصورة، تحتاج إلى تمرير صورة أولية إلى خط الأنابيب.

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png")
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline("Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", image=init_image).images[0]
```

</hfoption>
<hfoption id="specific pipeline">

يمكن تحميل نقاط التفتيش بواسطة فئة خط أنابيب محددة إذا كنت تعرفها بالفعل. على سبيل المثال، لتحميل نموذج Stable Diffusion، استخدم فئة [`StableDiffusionPipeline`].

```python
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
```

يمكن أيضًا استخدام نقطة التفتيش هذه لمهمة أخرى مثل الصورة إلى الصورة. للتمييز بين المهمة التي تريد استخدام نقطة التفتيش لها، يجب استخدام فئة خط أنابيب محددة للمهمة. على سبيل المثال، لاستخدام نقطة التفتيش نفسها للصورة إلى الصورة، استخدم فئة [`StableDiffusionImg2ImgPipeline`].

```py
from diffusers import StableDiffusionImg2ImgPipeline

pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
```

</hfoption>
</hfoptions>

استخدم المساحة أدناه لتقييم متطلبات ذاكرة خط الأنابيب قبل تنزيله وتحميله لمعرفة ما إذا كان يعمل على أجهزتك.

<div class="block dark:hidden">
<iframe
src="https://diffusers-compute-pipeline-size.hf.space?__theme=light"
width="850"
height="1600"
></iframe>
</div>
<div class="hidden dark:block">
<iframe
src="https://diffusers-compute-pipeline-size.hf.space?__theme=dark"
width="850"
height="1600"
></iframe>
</div>

### خط أنابيب محلي

لتحميل خط أنابيب محليًا، استخدم [git-lfs](https://git-lfs.github.com/) لتنزيل نقطة تفتيش يدويًا إلى القرص المحلي.

```bash
git-lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```

ينشئ هذا مجلدًا محليًا، ./stable-diffusion-v1-5، على القرص الخاص بك ويجب تمرير مساره إلى [`~DiffusionPipeline.from_pretrained`].

```python
from diffusers import DiffusionPipeline

stable_diffusion = DiffusionPipeline.from_pretrained("./stable-diffusion-v1-5", use_safetensors=True)
```

لن تقوم طريقة [`~DiffusionPipeline.from_pretrained`] بتنزيل الملفات من Hub عند اكتشاف مسار محلي، ولكن هذا يعني أيضًا أنها لن تقوم بتنزيل وتخزين أحدث التغييرات في نقطة التفتيش.

## تخصيص خط أنابيب

يمكنك تخصيص خط أنابيب عن طريق تحميل مكونات مختلفة فيه. هذا أمر مهم لأنك يمكن أن:

- قم بالتبديل إلى جدول زمني بسرعة إنشاء أسرع أو جودة إنشاء أعلى حسب احتياجاتك (استدعاء طريقة `scheduler.compatibles` على خط أنابيبك لمعرفة الجداول الزمنية المتوافقة)
- تغيير مكون خط أنابيب افتراضي إلى مكون أحدث وأفضل أداءً

على سبيل المثال، دعنا نخصص نقطة تفتيش [stabilityai/stable-diffusion-xl-base-1.0](https://hf.co/stabilityai/stable-diffusion-xl-base-1.0) الافتراضية باستخدام:

- [`HeunDiscreteScheduler`] لإنشاء صور عالية الجودة على حساب سرعة إنشاء أبطأ. يجب تمرير معلمة "subfolder="scheduler"" في [`~HeunDiscreteScheduler.from_pretrained`] لتحميل تكوين الجدول الزمني في المجلد الفرعي الصحيح [subfolder](https://hf.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main/scheduler) من مستودع خط الأنابيب.
- VAE أكثر استقرارًا يعمل في fp16.

```py
from diffusers import StableDiffusionXLPipeline, HeunDiscreteScheduler, AutoencoderKL
import torch

scheduler = HeunDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
```

الآن قم بتمرير الجدول الزمني الجديد وVAE إلى [`StableDiffusionXLPipeline`].

```py
pipeline = StableDiffusionXLPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
scheduler=scheduler,
vae=vae,
torch_dtype=torch.float16,
variant="fp16",
use_safetensors=True
).to("cuda")
```
## إعادة استخدام خط الأنابيب

عند تحميل خطوط أنابيب متعددة تشترك في نفس مكونات النموذج، من المنطقي إعادة استخدام المكونات المشتركة بدلاً من إعادة تحميل كل شيء في الذاكرة مرة أخرى، خاصة إذا كانت أجهزتك محدودة الذاكرة. على سبيل المثال:

1. لقد قمت بتوليد صورة باستخدام [`StableDiffusionPipeline`] ولكنك تريد تحسين جودتها باستخدام [`StableDiffusionSAGPipeline`]. حيث يتشارك خطا الأنابيب هذان في نفس النموذج المُدرب مسبقًا، لذا سيكون من هدر الذاكرة تحميل نفس النموذج مرتين.
2. تريد إضافة مكون نموذج، مثل [`MotionAdapter`](../api/pipelines/animatediff#animatediffpipeline)، إلى [`AnimateDiffPipeline`] الذي تم إنشاؤه من [`StableDiffusionPipeline`] موجود. مرة أخرى، نظرًا لأن خطي الأنابيب يتشاركان في نفس النموذج المُدرب مسبقًا، فسيكون من هدر الذاكرة تحميل خط أنابيب جديد تمامًا مرة أخرى.

مع واجهة برمجة التطبيقات (API) [`DiffusionPipeline.from_pipe`]، يمكنك التبديل بين خطوط أنابيب متعددة للاستفادة من ميزاتها المختلفة دون زيادة استخدام الذاكرة. إنه يشبه تشغيل ميزة وإيقافها في خط أنابيبك.

> [!TIP]
> للتبديل بين المهام (بدلاً من الميزات)، استخدم طريقة [`~DiffusionPipeline.from_pipe`] مع فئة [AutoPipeline](../api/pipelines/auto_pipeline) التي تحدد تلقائيًا فئة خط الأنابيب بناءً على المهمة (تعرف أكثر في البرنامج التعليمي [AutoPipeline](../tutorials/autopipeline)).

لنبدأ بـ [`StableDiffusionPipeline`] ثم نعيد استخدام مكونات النموذج المحملة لإنشاء [`StableDiffusionSAGPipeline`] لزيادة جودة التوليد. ستستخدم [`StableDiffusionPipeline`] مع [IP-Adapter](./ip_adapter) لتوليد دب يأكل البيتزا.

```python
from diffusers import DiffusionPipeline, StableDiffusionSAGPipeline
import torch
import gc
from diffusers.utils import load_image
from accelerate.utils import compute_module_sizes

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_neg_embed.png")

pipe_sd = DiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V6.0_B1_noVAE", torch_dtype=torch.float16)
pipe_sd.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe_sd.set_ip_adapter_scale(0.6)
pipe_sd.to("cuda")

generator = torch.Generator(device="cpu").manual_seed(33)
out_sd = pipe_sd(
prompt="bear eats pizza",
negative_prompt="wrong white balance, dark, sketches,worst quality,low quality",
ip_adapter_image=image,
num_inference_steps=50,
generator=generator,
).images[0]
out_sd
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/from_pipe_out_sd_0.png"/>
</div>

ولغرض المرجع، يمكنك التحقق من مقدار الذاكرة التي استهلكتها هذه العملية.

```python
def bytes_to_giga_bytes(bytes):
return bytes / 1024 / 1024 / 1024
print(f"Max memory allocated: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())} GB")
"Max memory allocated: 4.406213283538818 GB"
```

والآن، أعد استخدام نفس مكونات خط الأنابيب من [`StableDiffusionPipeline`] في [`StableDiffusionSAGPipeline`] باستخدام طريقة [`~DiffusionPipeline.from_pipe`].

> [!WARNING]
> قد لا تعمل بعض طرق خط الأنابيب بشكل صحيح على خطوط الأنابيب الجديدة التي تم إنشاؤها باستخدام [`~DiffusionPipeline.from_pipe`]. على سبيل المثال، تقوم طريقة [`~DiffusionPipeline.enable_model_cpu_offload`] بتثبيت الخطافات على مكونات النموذج بناءً على تسلسل تفريغ فريد لكل خط أنابيب. إذا تم تنفيذ النماذج بترتيب مختلف في خط الأنابيب الجديد، فقد لا يعمل التفريغ إلى وحدة المعالجة المركزية بشكل صحيح.
>
> لضمان عمل كل شيء كما هو متوقع، نوصي بإعادة تطبيق طريقة خط الأنابيب على خط أنابيب جديد تم إنشاؤه باستخدام [`~DiffusionPipeline.from_pipe`].

```python
pipe_sag = StableDiffusionSAGPipeline.from_pipe(
pipe_sd
)

generator = torch.Generator(device="cpu").manual_seed(33)
out_sag = pipe_sag(
prompt="bear eats pizza",
negative_prompt="wrong white balance, dark, sketches,worst quality,low quality",
ip_adapter_image=image,
num_inference_steps=50,
generator=generator,
guidance_scale=1.0,
sag_scale=0.75
).images[0]
out_sag
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/from_pipe_out_sag_1.png"/>
</div>

إذا قمت بالتحقق من استخدام الذاكرة، فستلاحظ أنها ظلت كما هي لأن [`StableDiffusionPipeline`] و [`StableDiffusionSAGPipeline`] يتشاركان في نفس مكونات خط الأنابيب. يسمح لك ذلك باستخدامها بشكل متبادل دون أي نفقات عامة إضافية للذاكرة.

```py
print(f"Max memory allocated: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())} GB")
"Max memory allocated: 4.406213283538818 GB"
```

دعنا نقوم بتحريك الصورة باستخدام [`AnimateDiffPipeline`] ونضيف أيضًا وحدة [`MotionAdapter`] إلى خط الأنابيب. وبالنسبة لـ [`AnimateDiffPipeline`]، يلزمك إلغاء تحميل محول IP أولاً وإعادة تحميله *بعد* إنشاء خط أنابيبك الجديد (ينطبق هذا فقط على [`AnimateDiffPipeline`]).

```py
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif

pipe_sag.unload_ip_adapter()
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)

pipe_animate = AnimateDiffPipeline.from_pipe(pipe_sd, motion_adapter=adapter)
pipe_animate.scheduler = DDIMScheduler.from_config(pipe_animate.scheduler.config, beta_schedule="linear")
# load IP-Adapter and LoRA weights again
pipe_animate.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe_animate.load_lora_weights("guoyww/animatediff-motion-lora-zoom-out", adapter_name="zoom-out")
pipe_animate.to("cuda")

generator = torch.Generator(device="cpu").manual_seed(33)
pipe_animate.set_adapters("zoom-out", adapter_weights=0.75)
out = pipe_animate(
prompt="bear eats pizza",
num_frames=16,
num_inference_steps=50,
ip_adapter_image=image,
generator=generator,
).frames[0]
export_to_gif(out, "out_animate.gif")
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/from_pipe_out_animate_3.gif"/>
</div>

يتطلب خط أنابيب [`AnimateDiffPipeline`] ذاكرة أكبر ويستهلك 15 جيجابايت من الذاكرة (راجع قسم [استخدام الذاكرة لـ from_pipe](#memory-usage-of-from_pipe) لمعرفة ما يعنيه ذلك بالنسبة لاستخدام الذاكرة).

```py
print(f"Max memory allocated: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())} GB")
"Max memory allocated: 15.178664207458496 GB"
```

### تعديل مكونات from_pipe

يمكن تخصيص خطوط الأنابيب المحملة باستخدام [`~DiffusionPipeline.from_pipe`] بمكونات نموذج أو طرق مختلفة. ومع ذلك، كلما قمت بتعديل حالة مكونات النموذج، فإنه يؤثر على جميع خطوط الأنابيب الأخرى التي تشترك في نفس المكونات. على سبيل المثال، إذا قمت بالاتصال بـ [`~diffusers.loaders.IPAdapterMixin.unload_ip_adapter`] على [`StableDiffusionSAGPipeline`]، فلن تتمكن من استخدام محول IP مع [`StableDiffusionPipeline`] لأنه تمت إزالته من مكوناتهم المشتركة.

```py
pipe.sag_unload_ip_adapter()

generator = torch.Generator(device="cpu").manual_seed(33)
out_sd = pipe_sd(
prompt="bear eats pizza",
negative_prompt="wrong white balance, dark, sketches,worst quality,low quality",
ip_adapter_image=image,
num_inference_steps=50,
generator=generator,
).images[0]
"AttributeError: 'NoneType' object has no attribute 'image_projection_layers'"
```

### استخدام الذاكرة لـ from_pipe

يتم تحديد متطلبات الذاكرة لتحميل خطوط أنابيب متعددة باستخدام [`~DiffusionPipeline.from_pipe`] بواسطة خط الأنابيب ذي أعلى استخدام للذاكرة بغض النظر عن عدد خطوط الأنابيب التي تقوم بإنشائها.

| خط الأنابيب | استخدام الذاكرة (جيجابايت) |
|---|---|
| StableDiffusionPipeline | 4.400 |
| StableDiffusionSAGPipeline | 4.400 |
| AnimateDiffPipeline | 15.178 |

نظرًا لأن خط أنابيب [`AnimateDiffPipeline`] لديه أعلى متطلبات الذاكرة، فإن "إجمالي استخدام الذاكرة" يعتمد فقط على خط أنابيب [`AnimateDiffPipeline`]. لن يزيد استخدام الذاكرة إذا قمت بإنشاء خطوط أنابيب إضافية طالما أن متطلبات الذاكرة الخاصة بها لا تتجاوز تلك الخاصة بخط أنابيب [`AnimateDiffPipeline`]. يمكن استخدام كل خط أنابيب بشكل متبادل دون أي نفقات عامة إضافية للذاكرة.

## فاحص الأمان

ينفذ برنامج Diffusers [فاحص أمان](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py) لنماذج Stable Diffusion التي يمكنها إنشاء محتوى ضار. يفحص فاحص الأمان الإخراج المولد مقابل محتوى غير آمن للعمل معروف ثابت مسبقًا. إذا كنت ترغب، لأي سبب من الأسباب، في تعطيل فاحص الأمان، قم بتمرير `safety_checker=None` إلى طريقة [`~DiffusionPipeline.from_pretrained`].

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None, use_safetensors=True)
"""
You have disabled the safety checker for <class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'> by passing `safety_checker=None`. Ensure that you abide by the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend keeping the safety filter enabled in all public-facing circumstances, disabling it only for use cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .
"""
```
## متغيرات نقطة التفتيش

متغير نقطة التفتيش هو عادةً نقطة تفتيش يتم تخزين أوزانها بشكل مختلف:

- يتم تخزين الأوزان في نوع نقطة عائمة مختلف، مثل [torch.float16](https://pytorch.org/docs/stable/tensors.html#data-types)، لأنها تتطلب نصف عرض النطاق الترددي والتخزين للتنزيل. لا يمكنك استخدام هذا المتغير إذا كنت تواصل التدريب أو تستخدم وحدة المعالجة المركزية.

- أوزان المتوسط غير الأسية (Non-exponential mean averaged) والتي لا يجب استخدامها للاستنتاج. يجب استخدام هذا المتغير لمواصلة الضبط الدقيق لنموذج.

> [!TIP]
> عندما تكون نقاط التفتيش لها هياكل نماذج متطابقة، ولكنها تم تدريبها على مجموعات بيانات مختلفة وبإعداد تدريب مختلف، فيجب تخزينها في مستودعات منفصلة. على سبيل المثال، [stabilityai/stable-diffusion-2](https://hf.co/stabilityai/stable-diffusion-2) و [stabilityai/stable-diffusion-2-1](https://hf.co/stabilityai/stable-diffusion-2-1) يتم تخزينها في مستودعات منفصلة.

وبخلاف ذلك، يكون المتغير **مطابقًا** تمامًا لنقطة التفتيش الأصلية. لديهم نفس تنسيق التسلسل تمامًا (مثل [safetensors](./using_safetensors))، وهيكل النموذج، ولديهم أشكال متطابقة للأنسجة.

| نوع نقطة التفتيش | اسم الوزن | الحجة لتحميل الأوزان |
| --- | --- | --- |
| الأصلي | diffusion_pytorch_model.safetensors | |
| نقطة عائمة | diffusion_pytorch_model.fp16.safetensors | `variant`، `torch_dtype` |
| غير EMA | diffusion_pytorch_model.non_ema.safetensors | `variant` |

هناك حجتان مهمتان لتحميل المتغيرات:

- `torch_dtype` يحدد دقة النقطة العائمة لنقطة التفتيش المحملة. على سبيل المثال، إذا كنت تريد توفير عرض النطاق الترددي عن طريق تحميل متغير fp16، فيجب عليك تعيين `variant="fp16"` و `torch_dtype=torch.float16` لتحويل الأوزان إلى fp16. وإلا، يتم تحويل أوزان fp16 إلى دقة fp32 الافتراضية.

إذا قمت بتعيين `torch_dtype=torch.float16` فقط، فسيتم أولاً تنزيل أوزان fp32 الافتراضية ثم تحويلها إلى fp16.

- `variant` يحدد أي ملفات يجب تحميلها من المستودع. على سبيل المثال، إذا كنت تريد تحميل متغير غير EMA لشبكة UNet من [runwayml/stable-diffusion-v1-5](https://hf.co/runwayml/stable-diffusion-v1-5/tree/main/unet)، فقم بتعيين `variant="non_ema"` لتنزيل ملف `non_ema`.

<hfoptions id="variants">
<hfoption id="fp16">

```py
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5"، variant="fp16"، torch_dtype=torch.float16، use_safetensors=True
)
```

</hfoption>
<hfoption id="non-EMA">

```py
pipeline = DiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5"، variant="non_ema"، use_safetensors=True
)
```

</hfoption>
</hfoptions>

استخدم معلمة `variant` في طريقة [`DiffusionPipeline.save_pretrained`] لحفظ نقطة تفتيش كنوع نقطة عائمة مختلف أو كمتغير غير EMA. يجب عليك محاولة حفظ متغير في نفس المجلد مثل نقطة التفتيش الأصلية، بحيث يكون لديك خيار تحميل كلاهما من نفس المجلد.

<hfoptions id="save">
<hfoption id="fp16">

```python
from diffusers import DiffusionPipeline

pipeline.save_pretrained("runwayml/stable-diffusion-v1-5"، variant="fp16")
```

</hfoption>
<hfoption id="non_ema">

```py
pipeline.save_pretrained("runwayml/stable-diffusion-v1-5"، variant="non_ema")
```

</hfoption>
</hfoptions>

إذا لم تقم بحفظ المتغير في مجلد موجود، فيجب عليك تحديد حجة `variant`؛ وإلا، فإنه سيرمي `Exception` لأنه لا يمكنه العثور على نقطة التفتيش الأصلية.

```python
# 👎 لن يعمل هذا
pipeline = DiffusionPipeline.from_pretrained(
"./stable-diffusion-v1-5"، torch_dtype=torch.float16، use_safetensors=True
)
# 👍 هذا يعمل
pipeline = DiffusionPipeline.from_pretrained(
"./stable-diffusion-v1-5"، variant="fp16"، torch_dtype=torch.float16، use_safetensors=True
)
```

## شرح DiffusionPipeline

كطريقة فئة، [DiffusionPipeline.from_pretrained] مسؤول عن شيئين:

- قم بتنزيل أحدث إصدار من هيكل المجلد المطلوب للاستدلال وتخزينه مؤقتًا. إذا كان أحدث هيكل مجلد متاحًا في ذاكرة التخزين المؤقت المحلية، فسيقوم [DiffusionPipeline.from_pretrained] بإعادة استخدام الذاكرة المؤقتة ولن يقوم بإعادة تنزيل الملفات.

- قم بتحميل الأوزان المخزنة مؤقتًا في خط الأنابيب الصحيح [class](../api/pipelines/overview#diffusers-summary) - المستردة من ملف `model_index.json` - وإرجاع مثيل منها.

يتطابق هيكل مجلد الأنابيب الأساسي مباشرةً مع مثيلات فئاتها. على سبيل المثال، يتطابق [`StableDiffusionPipeline`] مع هيكل المجلد في [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5).

```python
from diffusers import DiffusionPipeline

repo_id = "runwayml/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(repo_id، use_safetensors=True)
print(pipeline)
```

سترى أن الأنبوب هو مثيل من [`StableDiffusionPipeline`]، والذي يتكون من سبعة مكونات:

- `"feature_extractor"`: [`~transformers.CLIPImageProcessor`] من 🤗 Transformers.

- `"safety_checker"`: [مكون](https://github.com/huggingface/diffusers/blob/e55687e1e15407f60f32242027b7bb8170e58266/src/diffusers/pipelines/stable_diffusion/safety_checker.py#L32) لفحص المحتوى الضار.

- `"scheduler"`: مثيل من [`PNDMScheduler`].

- `"text_encoder"`: [`~transformers.CLIPTextModel`] من 🤗 Transformers.

- `"tokenizer"`: [`~transformers.CLIPTokenizer`] من 🤗 Transformers.

- `"unet"`: مثيل من [`UNet2DConditionModel`].

- `"vae"`: مثيل من [`AutoencoderKL`].

```json
StableDiffusionPipeline {
"feature_extractor": [
"transformers"،
"CLIPImageProcessor"
]،
"safety_checker": [
"stable_diffusion"،
"StableDiffusionSafetyChecker"
]،
"scheduler": [
"diffusers"،
"PNDMScheduler"
]،
"text_encoder": [
"transformers"،
"CLIPTextModel"
]،
"tokenizer": [
"transformers"،
"CLIPTokenizer"
]،
"unet": [
"diffusers"،
"UNet2DConditionModel"
]،
"vae": [
"diffusers"،
"AutoencoderKL"
]
}
```

قارن مكونات مثيل الأنبوب بهيكل مجلد [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main)، وسترى أنه يوجد مجلد منفصل لكل من المكونات في المستودع:

```
.
├── feature_extractor
│ └── preprocessor_config.json
├── model_index.json
├── safety_checker
│ ├── config.json
|   ├── model.fp16.safetensors
│   ├── model.safetensors
│   ├── pytorch_model.bin
|   └── pytorch_model.fp16.bin
├── scheduler
│ └── scheduler_config.json
├── text_encoder
│ ├── config.json
|   ├── model.fp16.safetensors
│   ├── model.safetensors
│   |── pytorch_model.bin
|   └── pytorch_model.fp16.bin
├── tokenizer
│ ├── merges.txt
│ ├── special_tokens_map.json
│ ├── tokenizer_config.json
│ └── vocab.json
├── unet
│ ├── config.json
│ ├── diffusion_pytorch_model.bin
|   |── diffusion_pytorch_model.fp16.bin
│   |── diffusion_pytorch_model.f16.safetensors
│   |── diffusion_pytorch_model.non_ema.bin
│   |── diffusion_pytorch_model.non_ema.safetensors
│   └── diffusion_pytorch_model.safetensors
|── vae
.   ├── config.json
.   ├── diffusion_pytorch_model.bin
├── diffusion_pytorch_model.fp16.bin
├── diffusion_pytorch_model.fp16.safetensors
└── diffusion_pytorch_model.safetensors
```

يمكنك الوصول إلى كل مكون من مكونات الأنبوب كسمة لعرض تكوينه:

```py
pipeline.tokenizer
CLIPTokenizer(
name_or_path="/root/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819/tokenizer"،
vocab_size=49408،
model_max_length=77،
is_fast=False،
padding_side="right"،
truncation_side="right"،
special_tokens={
"bos_token": AddedToken("<|startoftext|>"، rstrip=False، lstrip=False، single_word=False، normalized=True)،
"eos_token": AddedToken("<|endoftext|>"، rstrip=False، lstrip=False، single_word=False، normalized=True)،
"unk_token": AddedToken("<|endoftext|>"، rstrip=False، lstrip=False، single_word=False، normalized=True)،
"pad_token": "<|endoftext|>"،
}،
clean_up_tokenization_spaces=True
)
```

يتوقع كل خط أنابيب ملف [`model_index.json`](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/model_index.json) الذي يخبر [`DiffusionPipeline`]:

- أي فئة خط أنابيب لتحميلها من `_class_name`

- أي إصدار من 🧨 Diffusers تم استخدامه لإنشاء النموذج في `_diffusers_version`

- ما هي المكونات من أي مكتبة مخزنة في المجلدات الفرعية (`name` يقابل اسم المكون ومجلد الاسم، `library` يقابل اسم المكتبة لتحميل الفئة منها، و`class` يقابل اسم الفئة)

```json
{
"_class_name": "StableDiffusionPipeline"،
"_diffusers_version": "0.6.0"،
"feature_extractor": [
"transformers"،
"CLIPImageProcessor"
]،
"safety_checker": [
"stable_diffusion"،
"StableDiffusionSafetyChecker"
]،
"scheduler": [
"diffusers"،
"PNDMScheduler"
]،
"text_encoder": [
"transformers"،
"CLIPTextModel"
]،
"tokenizer": [
"transformers"،
"CLIPTokenizer"
]،
"unet": [
"diffusers"،
"UNet2DConditionModel"
]،
"vae": [
"diffusers"،
"AutoencoderKL"
]
}
```