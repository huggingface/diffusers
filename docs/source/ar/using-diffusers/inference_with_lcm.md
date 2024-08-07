# نموذج الاتساق الكامن

تمكّن نماذج الاتساق الكامنة (LCMs) من توليد صور عالية الجودة بسرعة من خلال التنبؤ المباشر بعملية الانتشار العكسي في الفراغ الكامن بدلاً من مساحة البكسل. وبعبارة أخرى، تحاول نماذج LCM التنبؤ بالصورة الخالية من الضوضاء من الصورة المشوشة، على عكس نماذج الانتشار النموذجية التي تزيل الضوضاء بشكل تكراري من الصورة المشوشة. ومن خلال تجنب عملية المعاينة التكرارية، يمكن لنماذج LCM توليد صور عالية الجودة في 2-4 خطوات بدلاً من 20-30 خطوة.

يتم استخلاص نماذج LCM من النماذج المُدربة مسبقًا والتي تتطلب حوالي 32 ساعة من الحوسبة على معالج A100. ولتسريع هذه العملية، تقوم LCM-LoRAs بتدريب مهايئ LoRA الذي يحتوي على عدد أقل بكثير من المعلمات التي يجب تدريبها مقارنة بالنموذج الكامل. يمكن توصيل LCM-LoRA بنموذج الانتشار بمجرد تدريبه.

سيوضح هذا الدليل كيفية استخدام نماذج LCM وLCM-LoRAs للاستدلال السريع على المهام، وكيفية استخدامها مع المهايئات الأخرى مثل ControlNet أو T2I-Adapter.

> [!TIP]
> تتوفر نماذج LCM وLCM-LoRAs لـ Stable Diffusion v1.5، وStable Diffusion XL، ونموذج SSD-1B. يمكنك العثور على نقاط التحقق الخاصة بها في مجموعة Latent Consistency.

## نص إلى صورة

<hfoptions id="lcm-text2img">
<hfoption id="LCM">  

لاستخدام نماذج LCM، تحتاج إلى تحميل نقطة التحقق LCM للنموذج المدعوم في [`UNet2DConditionModel`] واستبدال المخطط ب [`LCMScheduler`]. بعد ذلك، يمكنك استخدام الأنابيب كالمعتاد، وإمرار موجه النص لتوليد صورة في 4 خطوات فقط.

هناك بعض النقاط التي يجب مراعاتها عند استخدام نماذج LCM:

* عادة، يتم مضاعفة حجم الدفعة داخل الأنبوب للإرشاد الخالي من التصنيف. ولكن LCM يطبق الإرشاد مع ترميزات الإرشاد ولا يحتاج إلى مضاعفة حجم الدفعة، مما يؤدي إلى استدلال أسرع. الجانب السلبي هو أن المطالبات السلبية لا تعمل مع LCM لأنها لا تؤثر على عملية إزالة التشويش.
* النطاق المثالي لـ `guidance_scale` هو [3.، 13.] لأن هذا ما تم تدريب UNet عليه. ومع ذلك، فإن تعطيل `guidance_scale` بقيمة 1.0 فعال أيضًا في معظم الحالات.

```python
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, LCMScheduler
import torch

unet = UNet2DConditionModel.from_pretrained(
"latent-consistency/lcm-sdxl",
torch_dtype=torch.float16,
variant="fp16",
)
pipe = StableDiffusionXLPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16",
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
generator = torch.manual_seed(0)
image = pipe(
prompt=prompt, num_inference_steps=4, generator=generator, guidance_scale=8.0
).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_full_sdxl_t2i.png"/>
</div>

</hfoption>
<hfoption id="LCM-LoRA">  

لاستخدام LCM-LoRAs، تحتاج إلى استبدال المخطط ب [`LCMScheduler`] وتحميل أوزان LCM-LoRA باستخدام طريقة [`~loaders.LoraLoaderMixin.load_lora_weights`]. بعد ذلك، يمكنك استخدام الأنابيب كالمعتاد، وإمرار موجه النص لتوليد صورة في 4 خطوات فقط.

هناك بعض النقاط التي يجب مراعاتها عند استخدام LCM-LoRAs:

* عادة، يتم مضاعفة حجم الدفعة داخل الأنبوب للإرشاد الخالي من التصنيف. ولكن LCM يطبق الإرشاد مع ترميزات الإرشاد ولا يحتاج إلى مضاعفة حجم الدفعة، مما يؤدي إلى استدلال أسرع. الجانب السلبي هو أن المطالبات السلبية لا تعمل مع LCM لأنها لا تؤثر على عملية إزالة التشويش.
* يمكنك استخدام الإرشاد مع LCM-LoRAs، ولكنه حساس جدًا لقيم `guidance_scale` العالية ويمكن أن يؤدي إلى تشوهات في الصورة المولدة. أفضل القيم التي وجدناها هي بين [1.0، 2.0].
* استبدل [stabilityai/stable-diffusion-xl-base-1.0](https://hf.co/stabilityai/stable-diffusion-xl-base-1.0) بأي نموذج مُدرب بشكل دقيق. على سبيل المثال، جرب استخدام نقطة تفتيش [animagine-xl](https://huggingface.co/Linaqruf/animagine-xl) لتوليد صور أنيمي باستخدام SDXL.

```py
import torch
from diffusers import DiffusionPipeline, LCMScheduler

pipe = DiffusionPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0"،
variant="fp16"،
torch_dtype=torch.float16
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
generator = torch.manual_seed(42)
image = pipe(
prompt=prompt, num_inference_steps=4, generator=generator, guidance_scale=1.0
).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_sdxl_t2i.png"/>
</div>

</hfoption>
</hfoptions>

## صورة إلى صورة

<hfoptions id="lcm-img2img">
<hfoption id="LCM">  

لاستخدام نماذج LCM للصورة إلى صورة، تحتاج إلى تحميل نقطة التحقق LCM للنموذج المدعوم في [`UNet2DConditionModel`] واستبدال المخطط ب [`LCMScheduler`]. بعد ذلك، يمكنك استخدام الأنابيب كالمعتاد، وإمرار موجه النص والصورة الأولية لتوليد صورة في 4 خطوات فقط.

> [!TIP]
> جرب قيمًا مختلفة لـ `num_inference_steps`، و`strength`، و`guidance_scale` للحصول على أفضل النتائج.

```python
import torch
from diffusers import AutoPipelineForImage2Image, UNet2DConditionModel, LCMScheduler
from diffusers.utils import load_image

unet = UNet2DConditionModel.from_pretrained(
"SimianLuo/LCM_Dreamshaper_v7"،
subfolder="unet"،
torch_dtype=torch.float16،
)

pipe = AutoPipelineForImage2Image.from_pretrained(
"Lykon/dreamshaper-7"،
unet=unet،
torch_dtype=torch.float16،
variant="fp16"،
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png")
prompt = "Astronauts in a jungle, cold color palette, muted colors, detailed, 8k"
generator = torch.manual_seed(0)
image = pipe(
prompt،
image=init_image،
num_inference_steps=4،
guidance_scale=7.5،
strength=0.5،
generator=generator
).images[0]
image
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأولية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm-img2img.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة المولدة</figcaption>
</div>
</div>

</hfoption>
<hfoption id="LCM-LoRA">  

لاستخدام LCM-LoRAs للصورة إلى صورة، تحتاج إلى استبدال المخطط ب [`LCMScheduler`] وتحميل أوزان LCM-LoRA باستخدام طريقة [`~loaders.LoraLoaderMixin.load_lora_weights`]. بعد ذلك، يمكنك استخدام الأنابيب كالمعتاد، وإمرار موجه النص والصورة الأولية لتوليد صورة في 4 خطوات فقط.

> [!TIP]
> جرب قيمًا مختلفة لـ `num_inference_steps`، و`strength`، و`guidance_scale` للحصول على أفضل النتائج.

```py
import torch
from diffusers import AutoPipelineForImage2Image, LCMScheduler
from diffusers.utils import make_image_grid, load_image

pipe = AutoPipelineForImage2Image.from_pretrained(
"Lykon/dreamshaper-7"،
torch_dtype=torch.float16،
variant="fp16"،
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png")
prompt = "Astronauts in a jungle, cold color palette, muted colors, detailed, 8k"

generator = torch.manual_seed(0)
image = pipe(
prompt،
image=init_image،
num_inference_steps=4،
guidance_scale=1،
strength=0.6،
generator=generator
).images[0]
image
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأولية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm-lora-img2img.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة المولدة</figcaption>
</div>
</div>

</hfoption>
</hfoptions>

## Inpainting

لاستخدام LCM-LoRAs للحشو، تحتاج إلى استبدال المُجدول بـ [`LCMScheduler`] وتحميل أوزان LCM-LoRA باستخدام طريقة [`~loaders.LoraLoaderMixin.load_lora_weights`]. بعد ذلك، يمكنك استخدام الأنبوب كالمعتاد، وتمرير موجه نصي وصورة أولية وصورة قناع لتوليد صورة في 4 خطوات فقط.

```py
import torch
from diffusers import AutoPipelineForInpainting, LCMScheduler
from diffusers.utils import load_image, make_image_grid

pipe = AutoPipelineForInpainting.from_pretrained(
"runwayml/stable-diffusion-inpainting",
torch_dtype=torch.float16,
variant="fp16",
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
generator = torch.manual_seed(0)
image = pipe(
prompt=prompt,
image=init_image,
mask_image=mask_image,
generator=generator,
num_inference_steps=4,
guidance_scale=4,
).images[0]
image
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأولية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm-lora-inpaint.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة المولدة</figcaption>
</div>
</div>


## المحولات

تتوافق LCMs مع المحولات مثل LoRA وControlNet وT2I-Adapter وAnimateDiff. يمكنك جلب سرعة LCMs إلى هذه المحولات لتوليد الصور بأسلوب معين أو ضبط النموذج على إدخال آخر مثل صورة Canny.

### LoRA

يمكن ضبط محولات [LoRA](../using-diffusers/loading_adapters#lora) بسرعة لتعلم أسلوب جديد من بضع صور فقط وإضافتها إلى نموذج مدرب مسبقًا لتوليد الصور بهذا الأسلوب.

<hfoptions id="lcm-lora">
<hfoption id="LCM">

قم بتحميل نقطة تفتيش LCM للطراز المدعوم في [`UNet2DConditionModel`] واستبدل المُجدول بـ [`LCMScheduler`]. بعد ذلك، يمكنك استخدام طريقة [`~loaders.LoraLoaderMixin.load_lora_weights`] لتحميل أوزان LoRA في LCM وتوليد صورة ذات أسلوب في بضع خطوات.

```python
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, LCMScheduler
import torch

unet = UNet2DConditionModel.from_pretrained(
"latent-consistency/lcm-sdxl",
torch_dtype=torch.float16,
variant="fp16",
)
pipe = StableDiffusionXLPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16",
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("TheLastBen/Papercut_SDXL", weight_name="papercut.safetensors", adapter_name="papercut")

prompt = "papercut, a cute fox"
generator = torch.manual_seed(0)
image = pipe(
prompt=prompt, num_inference_steps=4, generator=generator, guidance_scale=8.0
).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_full_sdx_lora_mix.png"/>
</div>

</hfoption>
<hfoption id="LCM-LoRA">

استبدل المُجدول بـ [`LCMScheduler`]. بعد ذلك، يمكنك استخدام طريقة [`~loaders.LoraLoaderMixin.load_lora_weights`] لتحميل أوزان LCM-LoRA وأسلوب LoRA الذي تريد استخدامه. قم بدمج كلا محولَي LoRA باستخدام طريقة [`~loaders.UNet2DConditionLoadersMixin.set_adapters`] وقم بتوليد صورة ذات أسلوب في بضع خطوات.

```py
import torch
from diffusers import DiffusionPipeline, LCMScheduler

pipe = DiffusionPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
variant="fp16",
torch_dtype=torch.float16
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")
pipe.load_lora_weights("TheLastBen/Papercut_SDXL", weight_name="papercut.safetensors", adapter_name="papercut")

pipe.set_adapters(["lcm", "papercut"], adapter_weights=[1.0, 0.8])

prompt = "papercut, a cute fox"
generator = torch.manual_seed(0)
image = pipe(prompt, num_inference_steps=4, guidance_scale=1, generator=generator).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_sdx_lora_mix.png"/>
</div>

</hfoption>
</hfoptions>


### ControlNet

[ControlNet](./controlnet) هي محولات يمكن تدريبها على مجموعة متنوعة من الإدخالات مثل Canny edge أو تقدير الوضع أو العمق. يمكن إدراج ControlNet في الأنبوب لتوفير المزيد من الضبط والتحكم في النموذج من أجل التوليد الأكثر دقة.

يمكنك العثور على نماذج ControlNet الإضافية المدربة على إدخالات أخرى في مستودع [lllyasviel](https://hf.co/lllyasviel).

<hfoptions id="lcm-controlnet">
<hfoption id="LCM">

قم بتحميل نموذج ControlNet المدرب على صور Canny ومرره إلى [`ControlNetModel`]. بعد ذلك، يمكنك تحميل نموذج LCM في [`StableDiffusionControlNetPipeline`] واستبدال المُجدول بـ [`LCMScheduler`]. الآن، مرر صورة Canny إلى الأنبوب وقم بتوليد صورة.

> [!TIP]
> جرب قيمًا مختلفة لـ `num_inference_steps` و`controlnet_conditioning_scale` و`cross_attention_kwargs` و`guidance_scale` للحصول على أفضل النتائج.

```python
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
from diffusers.utils import load_image, make_image_grid

image = load_image(
"https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
).resize((512, 512))

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
"SimianLuo/LCM_Dreamshaper_v7",
controlnet=controlnet,
torch_dtype=torch.float16,
safety_checker=None,
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

generator = torch.manual_seed(0)
image = pipe(
"the mona lisa",
image=canny_image,
num_inference_steps=4,
generator=generator,
).images[0]
make_image_grid([canny_image, image], rows=1, cols=2)
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_full_sdv1-5_controlnet.png"/>
</div>

</hfoption>
<hfoption id="LCM-LoRA">

قم بتحميل نموذج ControlNet المدرب على صور Canny ومرره إلى [`ControlNetModel`]. بعد ذلك، يمكنك تحميل نموذج Stable Diffusion v1.5 في [`StableDiffusionControlNetPipeline`] واستبدال المُجدول بـ [`LCMScheduler`]. استخدم طريقة [`~loaders.LoraLoaderMixin.load_lora_weights`] لتحميل أوزان LCM-LoRA، ومرر صورة Canny إلى الأنبوب وقم بتوليد صورة.

> [!TIP]
> جرب قيمًا مختلفة لـ `num_inference_steps` و`controlnet_conditioning_scale` و`cross_attention_kwargs` و`guidance_scale` للحصول على أفضل النتائج.

```py
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
from diffusers.utils import load_image

image = load_image(
"https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
).resize((512, 512))

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5",
controlnet=controlnet,
torch_dtype=torch.float16,
safety_checker=None,
variant="fp16"
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

generator = torch.manual_seed(0)
image = pipe(
"the mona lisa",
image=canny_image,
num_inference_steps=4,
guidance_scale=1.5,
controlnet_conditioning_scale=0.8,
cross_attention_kwargs={"scale": 1},
generator=generator,
).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_sdv1-5_controlnet.png"/>
</div>

</hfoption>
</hfoptions>

### T2I-Adapter
[T2I-Adapter](./t2i_adapter) هو محول خفيف الوزن أكثر من ControlNet، يوفر مدخلاً إضافياً لتهيئة نموذج مُدرب مسبقًا. إنه أسرع من ControlNet ولكن النتائج قد تكون أسوأ قليلاً.
يمكنك العثور على نقاط تفتيش T2I-Adapter الإضافية المدربة على مدخلات أخرى في مستودع [TencentArc's](https://hf.co/TencentARC).

<hfoptions id="lcm-t2i">
<hfoption id="LCM">

قم بتحميل T2IAdapter الذي تم تدريبه على صور Canny ومرره إلى [`StableDiffusionXLAdapterPipeline`]. ثم قم بتحميل نقطة تفتيش LCM في [`UNet2DConditionModel`] واستبدل المخطط بـ [`LCMScheduler`]. الآن مرر صورة Canny إلى الأنبوب وقم بتوليد صورة.

```py
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import StableDiffusionXLAdapterPipeline, UNet2DConditionModel, T2IAdapter, LCMScheduler
from diffusers.utils import load_image, make_image_grid

# اكتشاف خريطة Canny في دقة منخفضة لتجنب التفاصيل عالية التردد
image = load_image(
"https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
).resize((384, 384))

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image).resize((1024, 1216))

adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-canny-sdxl-1.0", torch_dtype=torch.float16, varient="fp16").to("cuda")

unet = UNet2DConditionModel.from_pretrained(
"latent-consistency/lcm-sdxl",
torch_dtype=torch.float16,
variant="fp16",
)
pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
unet=unet,
adapter=adapter,
torch_dtype=torch.float16,
variant="fp16",
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

prompt = "the mona lisa, 4k picture, high quality"
negative_prompt = "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured"

generator = torch.manual_seed(0)
image = pipe(
prompt=prompt,
negative_prompt=negative_prompt,
image=canny_image,
num_inference_steps=4,
guidance_scale=5,
adapter_conditioning_scale=0.8,
adapter_conditioning_factor=1,
generator=generator,
).images[0]
```
<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm-t2i.png"/>
</div>
</hfoption>
<hfoption id="LCM-LoRA">

قم بتحميل T2IAdapter الذي تم تدريبه على صور Canny ومرره إلى [`StableDiffusionXLAdapterPipeline`]. استبدل المخطط بـ [`LCMScheduler`]، واستخدم طريقة [`~loaders.LoraLoaderMixin.load_lora_weights`] لتحميل أوزان LCM-LoRA. قم بتمرير صورة Canny إلى الأنبوب وقم بتوليد صورة.

```py
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import StableDiffusionXLAdapterPipeline, UNet2DConditionModel, T2IAdapter, LCMScheduler
from diffusers.utils import load_image, make_image_grid

# اكتشاف خريطة Canny في دقة منخفضة لتجنب التفاصيل عالية التردد
image = load_image(
"https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
).resize((384, 384))

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image).resize((1024, 1024))

adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-canny-sdxl-1.0", torch_dtype=torch.float16, varient="fp16").to("cuda")

pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
adapter=adapter,
torch_dtype=torch.float16,
variant="fp16",
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")

prompt = "the mona lisa, 4k picture, high quality"
negative_prompt = "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured"

generator = torch.manual_seed(0)
image = pipe(
prompt=prompt,
negative_prompt=negative_prompt,
image=canny_image,
num_inference_steps=4,
guidance_scale=1.5,
adapter_conditioning_scale=0.8,
adapter_conditioning_factor=1,
generator=generator,
).images[0]
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm-lora-t2i.png"/>
</div>
</hfoption>
</hfoptions>

### AnimateDiff
[AnimateDiff](../api/pipelines/animatediff) هو محول يضيف الحركة إلى صورة. يمكن استخدامه مع معظم نماذج Stable Diffusion، مما يحولها فعليًا إلى نماذج "توليد فيديو". يتطلب الحصول على نتائج جيدة باستخدام نموذج فيديو عادةً إنشاء عدة إطارات (16-24)، والتي يمكن أن تكون بطيئة جدًا باستخدام نموذج Stable Diffusion العادي. يمكن لـ LCM-LoRA تسريع هذه العملية عن طريق إجراء 4-8 خطوات فقط لكل إطار.
قم بتحميل [`AnimateDiffPipeline`] ومرر [`MotionAdapter`] إليه. ثم استبدل المخطط بـ [`LCMScheduler`]، واجمع كلا محولين LoRA باستخدام طريقة [`~loaders.UNet2DConditionLoadersMixin.set_adapters`]. الآن يمكنك تمرير موجه إلى الأنبوب وإنشاء صورة متحركة.

```py
import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler, LCMScheduler
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5")
pipe = AnimateDiffPipeline.from_pretrained(
"frankjoshua/toonyou_beta6",
motion_adapter=adapter,
).to("cuda")

# تعيين المخطط
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# تحميل LCM-LoRA
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="lcm")
pipe.load_lora_weights("guoyww/animatediff-motion-lora-zoom-in", weight_name="diffusion_pytorch_model.safetensors", adapter_name="motion-lora")

pipe.set_adapters(["lcm", "motion-lora"], adapter_weights=[0.55, 1.2])

prompt = "best quality, masterpiece, 1girl, looking at viewer, blurry background, upper body, contemporary, dress"
generator = torch.manual_seed(0)
frames = pipe(
prompt=prompt,
num_inference_steps=5,
guidance_scale=1.25,
cross_attention_kwargs={"scale": 1},
num_frames=24,
generator=generator
).frames[0]
export_to_gif(frames, "animation.gif")
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm-lora-animatediff.gif"/>
</div>