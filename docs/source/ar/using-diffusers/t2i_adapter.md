# T2I-Adapter
[T2I-Adapter](https://hf.co/papers/2302.08453) هو محول خفيف الوزن للتحكم في النماذج النصية-البصرية وتوفير إرشادات هيكلية أكثر دقة لها. يعمل عن طريق تعلم محاذاة بين المعرفة الداخلية للنموذج النصي-البصري وإشارة تحكم خارجية، مثل كشف الحواف أو تقدير العمق.

تصميم T2I-Adapter بسيط، حيث يتم تمرير الشرط إلى أربعة كتل لاستخلاص الخصائص وثلاثة كتل لتخفيض الدقة. وهذا يجعل تدريب محولات مختلفة للظروف المختلفة سريعًا وسهلاً، والتي يمكن توصيلها بالنموذج النصي-البصري. يشبه T2I-Adapter [ControlNet](controlnet) ولكنه أصغر (~77 مليون معامل) وأسرع لأنه يعمل مرة واحدة فقط أثناء عملية الانتشار. الجانب السلبي هو أن الأداء قد يكون أسوأ قليلاً من ControlNet.

سيوضح هذا الدليل كيفية استخدام T2I-Adapter مع نماذج Stable Diffusion المختلفة، وكيفية تكوين عدة محولات T2I-Adapter لفرض أكثر من شرط واحد.

> [!TIP]
> هناك العديد من محولات T2I-Adapter المتاحة لظروف مختلفة، مثل لوحة الألوان والعمق والرسوم التخطيطية ووضعية وتقسيم الصورة. يمكنك تجربتها من مستودع [TencentARC](https://hf.co/TencentARC).

قبل البدء، تأكد من تثبيت المكتبات التالية.

```py
# قم بإلغاء التعليق لتثبيت المكتبات الضرورية في Colab
#! pip install -q diffusers accelerate controlnet-aux==0.0.7
```

## نص إلى صورة

تعتمد النماذج النصية-البصرية على موجه لإنشاء صورة، ولكن في بعض الأحيان، قد لا يكون النص وحده كافيًا لتوفير إرشادات هيكلية أكثر دقة. يسمح T2I-Adapter بتوفير صورة تحكم إضافية لتوجيه عملية التوليد. على سبيل المثال، يمكنك توفير صورة Canny (حد أبيض لصورة على خلفية سوداء) لتوجيه النموذج لتوليد صورة ذات بنية مماثلة.

<hfoptions id="stablediffusion">
<hfoption id="Stable Diffusion 1.5">

قم بإنشاء صورة Canny باستخدام مكتبة [opencv-library](https://github.com/opencv/opencv-python).

```py
import cv2
import numpy as np
from PIL import Image
from diffusers.utils import load_image

image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png")
image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = Image.fromarray(image)
```

الآن قم بتحميل T2I-Adapter المشروط بصورة [Canny images](https://hf.co/TencentARC/t2iadapter_canny_sd15v2) ومرره إلى [`StableDiffusionAdapterPipeline`].

```py
import torch
from diffusers import StableDiffusionAdapterPipeline, T2IAdapter

adapter = T2IAdapter.from_pretrained("TencentARC/t2iadapter_canny_sd15v2", torch_dtype=torch.float16)
pipeline = StableDiffusionAdapterPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5",
adapter=adapter,
torch_dtype=torch.float16,
)
pipeline.to("cuda")
```

أخيرًا، قم بتمرير موجهك وصورة التحكم إلى الأنبوب.

```py
generator = torch.Generator("cuda").manual_seed(0)

image = pipeline(
prompt="صورة سينمائية لقطعة سجاد ناعمة وطرية على الطراز المتوسطي على أرضية خشبية، صورة فوتوغرافية 35 مم، فيلم، احترافية، بدقة 4K، عالية الدقة",
image=image,
generator=generator,
).images[0]
image
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/t2i-sd1.5.png"/>
</div>

</hfoption>

<hfoption id="Stable Diffusion XL">

قم بإنشاء صورة Canny باستخدام مكتبة [controlnet-aux](https://github.com/huggingface/controlnet_aux).

```py
from controlnet_aux.canny import CannyDetector
from diffusers.utils import load_image

canny_detector = CannyDetector()

image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png")
image = canny_detector(image, detect_resolution=384, image_resolution=1024)
```

الآن قم بتحميل T2I-Adapter المشروط بصورة [Canny images](https://hf.co/TencentARC/t2i-adapter-canny-sdxl-1.0) ومرره إلى [`StableDiffusionXLAdapterPipeline`].

```py
import torch
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL

scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-canny-sdxl-1.0", torch_dtype=torch.float16)
pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
adapter=adapter,
vae=vae,
scheduler=scheduler,
torch_dtype=torch.float16,
variant="fp16",
)
pipeline.to("cuda")
```

أخيرًا، قم بتمرير موجهك وصورة التحكم إلى الأنبوب.

```py
generator = torch.Generator("cuda").manual_seed(0)

image = pipeline(
prompt="صورة سينمائية لقطعة سجاد ناعمة وطرية على الطراز المتوسطي على أرضية خشبية، صورة فوتوغرافية 35 مم، فيلم، احترافية، بدقة 4K، عالية الدقة",
image=image,
generator=generator,
).images[0]
image
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/t2i-sdxl.png"/>
</div>

</hfoption>
</hfoptions>

## MultiAdapter

يمكن أيضًا تكوين محولات T2I-Adapter، مما يسمح باستخدام أكثر من محول واحد لفرض شروط تحكم متعددة على صورة. على سبيل المثال، يمكنك استخدام خريطة وضعية لتوفير التحكم الهيكلي وخريطة عمق للتحكم في العمق. يتم تمكين ذلك من خلال فئة [`MultiAdapter`].

دعونا نشترط نموذجًا نصيًا-بصريًا باستخدام محول وضعية وعمق. قم بإنشاء صورة العمق والوضعية ووضعها في قائمة.

```py
from diffusers.utils import load_image

pose_image = load_image(
"https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/keypose_sample_input.png"
)
depth_image = load_image(
"https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/depth_sample_input.png"
)
cond = [pose_image, depth_image]
prompt = ["Santa Claus walking into an office room with a beautiful city view"]
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/depth_sample_input.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">صورة العمق</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/keypose_sample_input.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">صورة الوضعية</figcaption>
</div>
</div>

قم بتحميل محولات الوضعية والعمق المقابلة كقائمة في فئة [`MultiAdapter`].

```py
import torch
from diffusers import StableDiffusionAdapterPipeline, MultiAdapter, T2IAdapter

adapters = MultiAdapter(
[
T2IAdapter.from_pretrained("TencentARC/t2iadapter_keypose_sd14v1"),
T2IAdapter.from_pretrained("TencentARC/t2iadapter_depth_sd14v1"),
]
)
adapters = adapters.to(torch.float16)
```

أخيرًا، قم بتحميل [`StableDiffusionAdapterPipeline`] بالمحولات، ومرر موجهك والصور المشروطة به. استخدم [`adapter_conditioning_scale`] لضبط وزن كل محول على الصورة.

```py
pipeline = StableDiffusionAdapterPipeline.from_pretrained(
"CompVis/stable-diffusion-v1-4",
torch_dtype=torch.float16,
adapter=adapters,
).to("cuda")

image = pipeline(prompt, cond, adapter_conditioning_scale=[0.7, 0.7]).images[0]
image
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/t2i-multi.png"/>
</div>