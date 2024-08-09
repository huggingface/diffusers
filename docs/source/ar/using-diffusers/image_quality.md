# التحكم في جودة الصورة

يمكن تحسين مكونات نموذج الانتشار، مثل UNet وscheduler، لتحسين جودة الصور المولدة مما يؤدي إلى تفاصيل أفضل. وتعد هذه التقنيات مفيدة بشكل خاص إذا لم تكن لديك الموارد اللازمة لمجرد استخدام نموذج أكبر للاستنتاج. يمكنك تمكين هذه التقنيات أثناء الاستنتاج دون أي تدريب إضافي.

سيوضح هذا الدليل كيفية تشغيل هذه التقنيات في خط أنابيب الخاص بك وكيفية تكوينها لتحسين جودة الصور التي تم إنشاؤها.

## التفاصيل

[FreeU](https://hf.co/papers/2309.11497) يحسن تفاصيل الصورة عن طريق إعادة توازن الأوزان الأساسية لشبكة U ونقاط الاتصال الخاصة بها. يمكن أن تتسبب نقاط الاتصال في تجاهل النموذج لبعض الدلالات الأساسية، مما قد يؤدي إلى تفاصيل غير طبيعية في الصورة المولدة. لا تتطلب هذه التقنية أي تدريب إضافي ويمكن تطبيقها أثناء التنقل أثناء الاستدلال لمهام مثل الصورة إلى الصورة والنص إلى الفيديو.

استخدم طريقة [~ pipelines.StableDiffusionMixin.enable_freeu] على خط أنابيب الخاص بك وقم بتكوين عوامل المقياس للعمود الفقري ('b1' و'b2') واتصالات التخطي ('s1' و's2'). يرمز الرقم بعد كل عامل مقياس إلى المرحلة في UNet حيث يتم تطبيق العامل. الق نظرة على مستودع [FreeU](https://github.com/ChenyangSi/FreeU#parameters) لمعرفة فرط المعلمات المرجعية لنماذج مختلفة.

<hfoptions id="freeu">

<hfoption id="Stable Diffusion v1-5">

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None
).to("cuda")
pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.5, b2=1.6)
generator = torch.Generator(device="cpu").manual_seed(33)
prompt = ""
image = pipeline(prompt, generator=generator).images[0]
image
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdv15-no-freeu.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">FreeU disabled</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdv15-freeu.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">FreeU enabled</figcaption>
</div>
</div>

</hfoption>

<hfoption id="Stable Diffusion v2-1">

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
"stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, safety_checker=None
).to("cuda")
pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.4, b2=1.6)
generator = torch.Generator(device="cpu").manual_seed(80)
prompt = "A squirrel eating a burger"
image = pipeline(prompt, generator=generator).images[0]
image
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdv21-no-freeu.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">FreeU disabled</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdv21-freeu.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">FreeU enabled</figcaption>
</div>
</div>

</hfoption>

<hfoption id="Stable Diffusion XL">

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
).to("cuda")
pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
generator = torch.Generator(device="cpu").manual_seed(13)
prompt = "A squirrel eating a burger"
image = pipeline(prompt, generator=generator).images[0]
image
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-no-freeu.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">FreeU disabled</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-freeu.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">FreeU enabled</figcaption>
</div>
</div>

</hfoption>

<hfoption id="Zeroscope">

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

pipeline = DiffusionPipeline.from_pretrained(
"damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16
).to("cuda")
# values come from https://github.com/lyn-rgb/FreeU_Diffusers#video-pipelines
pipeline.enable_freeu(b1=1.2, b2=1.4, s1=0.9, s2=0.2)
prompt = "Confident teddy bear surfer rides the wave in the tropics"
generator = torch.Generator(device="cpu").manual_seed(47)
video_frames = pipeline(prompt, generator=generator).frames[0]
export_to_video(video_frames, "teddy_bear.mp4", fps=10)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/video-no-freeu.gif"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">FreeU disabled</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/video-freeu.gif"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">FreeU enabled</figcaption>
</div>
</div>

</hfoption>

</hfoptions>

استدعاء طريقة [pipelines.StableDiffusionMixin.disable_freeu] لإلغاء تنشيط FreeU.

```py
pipeline.disable_freeu()
```