# ControlNet

ControlNet هو نوع من النماذج المستخدمة للتحكم في نماذج انتشار الصور من خلال توفير دخل إضافي للصورة. هناك العديد من أنواع المدخلات المشروطة (Canny edge، رسم المستخدم، وضع الإنسان، العمق، والمزيد) التي يمكنك استخدامها للتحكم في نموذج الانتشار. وهذا مفيد للغاية لأنه يمنحك تحكمًا أكبر في إنشاء الصور، مما يسهل إنشاء صور محددة دون الحاجة إلى تجربة مطالبات نصية أو قيم إزالة الضوضاء المختلفة.

<Tip>
اطلع على القسم 3.5 من ورقة [ControlNet](https://huggingface.co/papers/2302.05543) الإصدار 1 للحصول على قائمة بتنفيذات ControlNet لمختلف المدخلات المشروطة. يمكنك العثور على نماذج ControlNet المشروطة الرسمية والمستقرة على ملف تعريف [lllyasviel](https://huggingface.co/lllyasviel) Hub، والمزيد من النماذج [المدربة من قبل المجتمع](https://huggingface.co/models?other=stable-diffusion&other=controlnet) على Hub.

بالنسبة لنماذج ControlNet SDXL (Stable Diffusion XL)، يمكنك العثور عليها في منظمة 🤗 [Diffusers](https://huggingface.co/diffusers) Hub، أو يمكنك تصفح النماذج [المدربة من قبل المجتمع](https://huggingface.co/models?other=stable-diffusion-xl&other=controlnet) على Hub.
</Tip>

يحتوي نموذج ControlNet على مجموعتين من الأوزان (أو الكتل) متصلة بطبقة التصفية الصفرية:

- *نسخة محمية* تحافظ على كل ما تعلمه نموذج الانتشار المسبق الكبير
- *نسخة قابلة للتدريب* يتم تدريبها على إدخال الشرط الإضافي

نظرًا لأن النسخة المحمية تحافظ على النموذج المسبق التدريب، فإن تدريب وتنفيذ ControlNet على إدخال شرط جديد سريع مثل ضبط نموذج آخر لأنك لا تدرب النموذج من الصفر.

سيوضح هذا الدليل كيفية استخدام ControlNet للتحويل من نص إلى صورة، ومن صورة إلى صورة، والطلاء التلقائي، والمزيد! هناك العديد من أنواع مدخلات ControlNet للاختيار من بينها، ولكن في هذا الدليل، سنركز فقط على بعض منها. لا تتردد في تجربة مدخلات الشرط الأخرى!

قبل البدء، تأكد من تثبيت المكتبات التالية:

```py
# قم بإلغاء التعليق لتثبيت المكتبات الضرورية في Colab
#! pip install -q diffusers transformers accelerate opencv-python
```

## من نص إلى صورة

بالنسبة للتحويل من نص إلى صورة، عادةً ما يتم تمرير مطالبة نصية إلى النموذج. ولكن مع ControlNet، يمكنك تحديد إدخال شرط إضافي. دعونا نشترط النموذج مع صورة Canny، وهو مخطط أبيض لصورة على خلفية سوداء. بهذه الطريقة، يمكن لـ ControlNet استخدام صورة Canny كعنصر تحكم لتوجيه النموذج لإنشاء صورة بنفس المخطط.

قم بتحميل صورة واستخدم مكتبة [opencv-python](https://github.com/opencv/opencv-python) لاستخراج صورة Canny:

```py
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np

original_image = load_image(
"https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)

image = np.array(original_image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأصلية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/vermeer_canny_edged.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">صورة Canny</figcaption>
</div>
</div>

بعد ذلك، قم بتحميل نموذج ControlNet المشروط على اكتشاف حافة Canny ومرره إلى [`StableDiffusionControlNetPipeline`]. استخدم [`UniPCMultistepScheduler`] الأسرع وقم بتمكين نقل النموذج إلى وحدة المعالجة المركزية للتسريع الاستدلال وتقليل استخدام الذاكرة.

```py
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5"، controlnet=controlnet، torch_dtype=torch.float16، use_safetensors=True
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
```

الآن قم بتمرير مطالبتك وصورة Canny إلى الأنبوب:

```py
output = pipe(
"the mona lisa"، image=canny_image
).images[0]
make_image_grid([original_image, canny_image, output], rows=1, cols=3)
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-text2img.png"/>
</div>

## من صورة إلى صورة

بالنسبة للتحويل من صورة إلى صورة، عادةً ما يتم تمرير صورة أولية وطلب إلى الأنبوب لإنشاء صورة جديدة. مع ControlNet، يمكنك تمرير إدخال شرط إضافي لتوجيه النموذج. دعونا نشترط النموذج باستخدام خريطة العمق، وهي صورة تحتوي على معلومات مكانية. بهذه الطريقة، يمكن لـ ControlNet استخدام خريطة العمق كعنصر تحكم لتوجيه النموذج لإنشاء صورة تحافظ على المعلومات المكانية.

ستستخدم [`StableDiffusionControlNetImg2ImgPipeline`] لهذه المهمة، والتي تختلف عن [`StableDiffusionControlNetPipeline`] لأنها تسمح لك بتمرير صورة أولية كنقطة بداية لعملية إنشاء الصورة.

قم بتحميل صورة واستخدم خط أنابيب `depth-estimation` [`~transformers.Pipeline`] من 🤗 Transformers لاستخراج خريطة عمق الصورة:

```py
import torch
import numpy as np

from transformers import pipeline
from diffusers.utils import load_image, make_image_grid

image = load_image(
"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-img2img.jpg"
)

def get_depth_map(image, depth_estimator):
image = depth_estimator(image)["depth"]
image = np.array(image)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
detected_map = torch.from_numpy(image).float() / 255.0
depth_map = detected_map.permute(2, 0, 1)
return depth_map

depth_estimator = pipeline("depth-estimation")
depth_map = get_depth_map(image, depth_estimator).unsqueeze(0).half().to("cuda")
```

بعد ذلك، قم بتحميل نموذج ControlNet المشروط على خرائط العمق ومرره إلى [`StableDiffusionControlNetImg2ImgPipeline`]. استخدم [`UniPCMultistepScheduler`] الأسرع وقم بتمكين نقل النموذج إلى وحدة المعالجة المركزية للتسريع الاستدلال وتقليل استخدام الذاكرة.

```py
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth"، torch_dtype=torch.float16، use_safetensors=True)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5"، controlnet=controlnet، torch_dtype=torch.float16، use_safetensors=True
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
```

الآن قم بتمرير مطالبتك، والصورة الأولية، وخريطة العمق إلى الأنبوب:

```py
output = pipe(
"lego batman and robin"، image=image, control_image=depth_map,
).images[0]
make_image_grid([image, output], rows=1, cols=2)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-img2img.jpg"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأصلية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-img2img-2.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة المولدة</figcaption>
</div>
</div>

## Inpainting

بالنسبة لتقنية Inpainting، فأنت بحاجة إلى صورة أولية، وصورة قناع، ووصف يحدد ما يجب استبدال القناع به. تسمح نماذج ControlNet بإضافة صورة تحكم أخرى لتهيئة النموذج. دعنا نقم بتهيئة النموذج باستخدام قناع Inpainting. بهذه الطريقة، يمكن لـ ControlNet استخدام قناع Inpainting كوسيلة تحكم لتوجيه النموذج لتوليد صورة داخل منطقة القناع.

قم بتحميل صورة أولية وصورة قناع:

```py
from diffusers.utils import load_image, make_image_grid

init_image = load_image(
"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-inpaint.jpg"
)
init_image = init_image.resize((512, 512))

mask_image = load_image(
"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-inpaint-mask.jpg"
)
mask_image = mask_image.resize((512, 512))
make_image_grid([init_image, mask_image], rows=1, cols=2)
```

قم بإنشاء دالة لإعداد صورة التحكم من الصورة الأولية وصورة القناع. سيؤدي هذا إلى إنشاء مصفوفة لتحديد البكسلات في `init_image` كبكسلات مقنعة إذا كان البكسل المقابل في `mask_image` أعلى من عتبة معينة.

```py
import numpy as np
import torch

def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1]
    image[image_mask > 0.5] = -1.0 # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

control_image = make_inpaint_condition(init_image, mask_image)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-inpaint.jpg"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأصلية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-inpaint-mask.jpg"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">صورة القناع</figcaption>
</div>
</div>

قم بتحميل نموذج ControlNet المشروط بـ Inpainting ومرره إلى [`StableDiffusionControlNetInpaintPipeline`]. استخدم [`UniPCMultistepScheduler`] الأسرع وقم بتمكين تفريغ النموذج لتسريع الاستنتاج وتقليل استخدام الذاكرة.

```py
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
```

الآن قم بتمرير وصفك، والصورة الأولية، وصورة القناع، وصورة التحكم إلى خط الأنابيب:

```py
output = pipe(
"corgi face with large ears, detailed, pixar, animated, disney",
num_inference_steps=20,
eta=1.0,
image=init_image,
mask_image=mask_image,
control_image=control_image,
).images[0]
make_image_grid([init_image, mask_image, output], rows=1, cols=3)
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-inpaint-result.png"/>
</div>

## وضع التخمين

[وضع التخمين](https://github.com/lllyasviel/ControlNet/discussions/188) لا يتطلب توفير وصف لشبكة التحكم على الإطلاق! وهذا يجبر مشفر ControlNet على بذل قصارى جهده لـ "تخمين" محتويات خريطة التحكم المدخلة (خريطة العمق، تقدير الوضع، Canny edge، إلخ).

يقوم وضع التخمين بتعديل مقياس المخلفات الناتجة عن ControlNet وفقًا لنسبة ثابتة تعتمد على عمق الكتلة. يقابل الكتلة الأعمق `DownBlock` 0.1، ومع زيادة عمق الكتل، يزيد المقياس بشكل أسّي بحيث يصبح مقياس إخراج `MidBlock` 1.0.

<Tip>
لا يؤثر وضع التخمين على تهيئة الوصف ويمكنك لا تزال توفير وصف إذا أردت ذلك.
</Tip>

قم بتعيين `guess_mode=True` في خط الأنابيب، ومن [المستحسن](https://github.com/lllyasviel/ControlNet#guess-mode--non-prompt-mode) تعيين قيمة `guidance_scale` بين 3.0 و5.0.

```py
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image, make_image_grid
import numpy as np
import torch
from PIL import Image
import cv2

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, use_safetensors=True).to("cuda")

original_image = load_image("https://huggingface.co/takuma104/controlnet_dev/resolve/main/bird_512x512.png")

image = np.array(original_image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

image = pipe("", image=canny_image, guess_mode=True, guidance_scale=3.0).images[0]
make_image_grid([original_image, canny_image, image], rows=1, cols=3)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare_guess_mode/output_images/diffusers/output_bird_canny_0.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الوضع العادي مع الوصف</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare_guess_mode/output_images/diffusers/output_bird_canny_0_gm.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">وضع التخمين بدون وصف</figcaption>
</div>
</div>

## ControlNet with Stable Diffusion XL

في الوقت الحالي، لا يوجد الكثير من نماذج ControlNet المتوافقة مع Stable Diffusion XL (SDXL)، ولكننا قمنا بتدريب نموذجين كاملين من نماذج ControlNet المتوافقة مع SDXL المشروطة على كشف حواف كاني (Canny edge detection) وخرائط العمق (depth maps). كما نجري تجارب لإنشاء إصدارات أصغر من نماذج ControlNet المتوافقة مع SDXL لتسهيل تشغيلها على الأجهزة المحدودة الموارد. يمكنك العثور على هذه النقاط المرجعية على [منظمة 🤗 Diffusers Hub](https://huggingface.co/diffusers)!

لنستخدم نموذج SDXL ControlNet المشروط على صور كاني لتوليد صورة. ابدأ بتحميل صورة وإعداد صورة كاني:

```py
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np
import torch

original_image = load_image(
"https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
)

image = np.array(original_image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
make_image_grid([original_image, canny_image], rows=1, cols=2)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأصلية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hf-logo-canny.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">صورة كاني</figcaption>
</div>
</div>

قم بتحميل نموذج SDXL ControlNet المشروط على كشف حواف كاني ومرره إلى [`StableDiffusionXLControlNetPipeline`]. يمكنك أيضًا تمكين نقل النموذج إلى وحدة المعالجة المركزية (CPU) لتقليل استخدام الذاكرة.

```py
controlnet = ControlNetModel.from_pretrained(
"diffusers/controlnet-canny-sdxl-1.0",
torch_dtype=torch.float16,
use_safetensors=True
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
controlnet=controlnet,
vae=vae,
torch_dtype=torch.float16,
use_safetensors=True
)
pipe.enable_model_cpu_offload()
```

الآن، قم بتمرير المحث (prompt) (والمحث السلبي إن كنت تستخدمه) وصورة كاني إلى خط الأنابيب (pipeline):

<Tip>

يحدد معامل [`controlnet_conditioning_scale`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetPipeline.__call__.controlnet_conditioning_scale) مقدار الوزن المخصص لمدخلات التكييف. القيمة الموصى بها هي 0.5 لتحقيق تعميم جيد، ولكن يمكنك تجربة أرقام أخرى!

</Tip>

```py
prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
negative_prompt = 'low quality, bad quality, sketches'

image = pipe(
prompt,
negative_prompt=negative_prompt,
image=canny_image,
controlnet_conditioning_scale=0.5,
).images[0]
make_image_grid([original_image, canny_image, image], rows=1, cols=3)
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/resolve/main/out_hug_lab_7.png"/>
</div>

يمكنك أيضًا استخدام [`StableDiffusionXLControlNetPipeline`] في وضع التخمين عن طريق تعيين المعامل إلى `True`:

```py
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
import numpy as np
import torch
import cv2
from PIL import Image

prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
negative_prompt = "low quality, bad quality, sketches"

original_image = load_image(
"https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
)

controlnet = ControlNetModel.from_pretrained(
"diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16, use_safetensors=True
)
pipe.enable_model_cpu_offload()

image = np.array(original_image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

image = pipe(
prompt, negative_prompt=negative_prompt, controlnet_conditioning_scale=0.5, image=canny_image, guess_mode=True,
).images[0]
make_image_grid([original_image, canny_image, image], rows=1, cols=3)
```

<Tip>

يمكنك استخدام نموذج تحسين (refiner model) مع `StableDiffusionXLControlNetPipeline` لتحسين جودة الصورة، تمامًا كما تفعل مع `StableDiffusionXLPipeline` العادي.

راجع قسم [تحسين جودة الصورة](./sdxl#refine-image-quality) لمعرفة كيفية استخدام نموذج التحسين.

تأكد من استخدام `StableDiffusionXLControlNetPipeline` ومرر `image` و`controlnet_conditioning_scale`.

```py
base = StableDiffusionXLControlNetPipeline(...)
image = base(
prompt=prompt,
controlnet_conditioning_scale=0.5,
image=canny_image,
num_inference_steps=40,
denoising_end=0.8,
output_type="latent",
).images
# الباقي كما هو تمامًا مع StableDiffusionXLPipeline
```

</Tip>


## MultiControlNet 

يمكنك تكوين العديد من عمليات ضبط ControlNet من مدخلات الصور المختلفة لإنشاء *MultiControlNet*. وللحصول على نتائج أفضل، من المفيد غالبًا:

1. قناع الضبط بحيث لا تتداخل (على سبيل المثال، قناع منطقة صورة Canny حيث يقع ضبط الوضع)
2. تجربة مع معلمة [`controlnet_conditioning_scale`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetPipeline.__call__.controlnet_conditioning_scale) لتحديد مقدار الوزن الذي يجب تعيينه لكل إدخال ضبط

في هذا المثال، ستجمع بين صورة Canny وصورة تقدير وضع الإنسان لإنشاء صورة جديدة.

قم بإعداد ضبط صورة Canny:

```py
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import numpy as np
import cv2

original_image = load_image(
"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/landscape.png"
)
image = np.array(original_image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)

# zero out middle columns of image where pose will be overlaid
zero_start = image.shape[1] // 4
zero_end = zero_start + image.shape[1] // 2
image[:, zero_start:zero_end] = 0

image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
make_image_grid([original_image, canny_image], rows=1, cols=2)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/landscape.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأصلية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/controlnet/landscape_canny_masked.png"/>
<figcaption class="mt-ompi text-center text-sm text-gray-500">صورة Canny</figcaption>
</div>
</div>

بالنسبة لتقدير الوضع البشري، قم بتثبيت [controlnet_aux](https://github.com/patrickvonplaten/controlnet_aux):

```py
# uncomment to install the necessary library in Colab
#! pip install -q controlnet-aux
```

قم بإعداد ضبط تقدير الوضع البشري:

```py
from controlnet_aux import OpenposeDetector

openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
original_image = load_image(
"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png"
)
openpose_image = openpose(original_image)
make_image_grid([original_image, openpose_image], rows=1, cols=2)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأصلية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/controlnet/person_pose.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">صورة الوضع البشري</figcaption>
</div>
</div>

قم بتحميل قائمة نماذج ControlNet التي تتوافق مع كل ضبط، ومررها إلى [`StableDiffusionXLControlNetPipeline`]. استخدم [`UniPCMultistepScheduler`] الأسرع وقم بتمكين تفريغ النموذج لتقليل استخدام الذاكرة.

```py
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL, UniPCMultistepScheduler
import torch

controlnets = [
ControlNetModel.from_pretrained(
"thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16
),
ControlNetModel.from_pretrained(
"diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True
),
]

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnets, vae=vae, torch_dtype=torch.float16, use_safetensors=True
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
```

الآن يمكنك تمرير مطالبتك (مطالبة سلبية إذا كنت تستخدم واحدة)، وصورة Canny، وصورة الوضع إلى الأنبوب:

```py
prompt = "a giant standing in a fantasy landscape, best quality"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

generator = torch.manual_seed(1)

images = [openpose_image.resize((1024, 1024)), canny_image.resize((1024, 1024))]

images = pipe(
prompt,
image=images,
num_inference_steps=25,
generator=generator,
negative_prompt=negative_prompt,
num_images_per_prompt=3,
controlnet_conditioning_scale=[1.0, 0.8],
).images
make_image_grid([original_image, canny_image, openpose_image,
images[0].resize((512, 512)), images[1].resize((512, 512)), images[2].resize((512, 512))], rows=2, cols=3)
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/multicontrolnet.png"/>
</div>