# Stable Diffusion XL

[[open-in-colab]]

[Stable Diffusion XL](https://huggingface.co/papers/2307.01952) (SDXL) هو نموذج قوي لتوليد الصور النصية الذي يبني على النماذج السابقة لستيبل ديفيشن بثلاث طرق رئيسية:

1. الشبكة U أكبر بثلاث مرات، وSDXL تجمع بين مشفر نص ثانٍ (OpenCLIP ViT-bigG/14) مع المشفر النصي الأصلي لزيادة عدد المعلمات بشكل كبير.
2. يقدم التكييف حسب الحجم والمحاصيل للحفاظ على بيانات التدريب من أن يتم تجاهلها والحصول على مزيد من التحكم في كيفية اقتصاص الصورة المولدة.
3. يقدم عملية نموذج من مرحلتين؛ النموذج "الأساسي" (يمكن تشغيله أيضًا كنموذج مستقل) يقوم بتوليد صورة كمدخلات لنموذج "التحسين" الذي يضيف تفاصيل عالية الجودة.

سيوضح هذا الدليل كيفية استخدام SDXL للصور النصية، والصور للصور، والتحسين.

قبل البدء، تأكد من تثبيت المكتبات التالية:

```py
# قم بإلغاء التعليق لتثبيت المكتبات الضرورية في Colab
#! pip install -q diffusers transformers accelerate invisible-watermark>=0.2.0
```

<Tip warning={true}>
نوصي بتثبيت مكتبة [invisible-watermark](https://pypi.org/project/invisible-watermark/) للمساعدة في تحديد الصور التي تم توليدها. إذا تم تثبيت مكتبة invisible-watermark، فسيتم استخدامها بشكل افتراضي. لإيقاف تشغيل أداة watermarker:

```py
pipeline = StableDiffusionXLPipeline.from_pretrained(..., add_watermarker=False)
```

</Tip>

## تحميل نقاط تفتيش النموذج

قد يتم تخزين أوزان النموذج في مجلدات فرعية منفصلة على Hub أو محليًا، وفي هذه الحالة، يجب استخدام طريقة [`~StableDiffusionXLPipeline.from_pretrained`] :

```py
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch

pipeline = StableDiffusionXLPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
).to("cuda")
```

يمكنك أيضًا استخدام طريقة [`~StableDiffusionXLPipeline.from_single_file`] لتحميل نقطة تفتيش النموذج المخزنة بتنسيق ملف واحد (`.ckpt` أو `.safetensors`) من Hub أو محليًا:

```py
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch

pipeline = StableDiffusionXLPipeline.from_single_file(
"https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors",
torch_dtype=torch.float16
).to("cuda")

refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(
"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/sd_xl_refiner_1.0.safetensors", torch_dtype=torch.float16
).to("cuda")
```

## نص إلى صورة

بالنسبة للنص إلى الصورة، قم بتمرير موجه نصي. بشكل افتراضي، يقوم SDXL بتوليد صورة 1024x1024 للحصول على أفضل النتائج. يمكنك تجربة تعيين معلمات `height` و`width` إلى 768x768 أو 512x512، ولكن أي شيء أقل من 512x512 من غير المرجح أن يعمل.

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "رائد فضاء في الغابة، لوحة ألوان باردة، ألوان خافتة، مفصلة، 8k"
image = pipeline_text2image(prompt=prompt).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png" alt="صورة مُنشأة لرائد فضاء في الغابة" />
</div>

## صورة إلى صورة

بالنسبة للصورة إلى الصورة، يعمل SDXL بشكل جيد خاصة مع أحجام الصور بين 768x768 و1024x1024. قم بتمرير صورة أولية، وموجه نصي لتحديد الصورة به:

```py
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

# استخدم from_pipe لتجنب استهلاك ذاكرة إضافية عند تحميل نقطة تفتيش
pipeline = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to("cuda")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
init_image = load_image(url)
prompt = "كلب يمسك قرص طائر في الغابة"
image = pipeline(prompt، image=init_image، strength=0.8، guidance_scale=10.5).images[0]
make_image_grid([init_image، image]، rows=1، cols=2)
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-img2img.png" alt="صورة مُنشأة لكلب يمسك قرصًا طائرًا في الغابة" />
</div>

## التحسين

بالنسبة للتحسين، ستحتاج إلى الصورة الأصلية وقناع لما تريد استبداله في الصورة الأصلية. قم بإنشاء موجه لوصف ما تريد استبدال المنطقة المقنعة به.

```py
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

# استخدم from_pipe لتجنب استهلاك ذاكرة إضافية عند تحميل نقطة تفتيش
pipeline = AutoPipelineForInpainting.from_pipe(pipeline_text2image).to("cuda")

img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
mask_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-inpaint-mask.png"

init_image = load_image(img_url)
mask_image = load_image(mask_url)

prompt = "غواص في البحر العميق يطفو"
image = pipeline(prompt=prompt، image=init_image، mask_image=mask_image، strength=0.85، guidance_scale=12.5).images[0]
make_image_grid([init_image، mask_image، image]، rows=1، cols=3)
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-inpaint.png" alt="صورة مُنشأة لغواص في البحر العميق في الغابة" />
</div>

## تحسين جودة الصورة

يتضمن SDXL [نموذج التحسين](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0) المتخصص في إزالة تشويش صور مرحلة الضوضاء المنخفضة لتوليد صور عالية الجودة من النموذج الأساسي. هناك طريقتان لاستخدام المحسن:

1. استخدام النموذج الأساسي ونموذج التحسين معًا لإنتاج صورة محسنة
2. استخدام النموذج الأساسي لإنتاج صورة، ثم استخدام نموذج التحسين لإضافة المزيد من التفاصيل إلى الصورة (هكذا تم تدريب SDXL في الأصل)

### النموذج الأساسي + نموذج التحسين

عندما تستخدم النموذج الأساسي ونموذج التحسين معًا لتوليد صورة، يُعرف ذلك باسم ["مجموعة من خبراء إزالة التشويش"](https://research.nvidia.com/labs/dir/eDiff-I/). يتطلب نهج مجموعة الخبراء في إزالة التشويش عددًا أقل من خطوات إزالة التشويش الإجمالية مقابل تمرير إخراج النموذج الأساسي إلى نموذج التحسين، لذا يجب أن يكون أسرع بكثير. ومع ذلك، فلن تتمكن من فحص إخراج النموذج الأساسي لأنه لا يزال يحتوي على قدر كبير من الضوضاء.

باعتباره مجموعة من خبراء إزالة التشويش، يعمل النموذج الأساسي كخبير خلال مرحلة انتشار الضوضاء العالية ويعمل نموذج التحسين كخبير خلال مرحلة انتشار الضوضاء المنخفضة. قم بتحميل النموذج الأساسي ونموذج التحسين:

```py
from diffusers import DiffusionPipeline
import torch

base = DiffusionPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0"، torch_dtype=torch.float16، variant="fp16"، use_safetensors=True
).to("cuda")

refiner = DiffusionPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-refiner-1.0"،
text_encoder_2=base.text_encoder_2،
vae=base.vae،
torch_dtype=torch.float16،
use_safetensors=True،
variant="fp16"،
).to("cuda")
```

لاستخدام هذا النهج، تحتاج إلى تحديد عدد خطوات الوقت لكل نموذج للعمل خلال مراحلها الخاصة. بالنسبة للنموذج الأساسي، يتم التحكم فيه بواسطة معلمة [`denoising_end`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline.__call__.denoising_end) ولنموذج التحسين، يتم التحكم فيه بواسطة معلمة [`denoising_start`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLImg2ImgPipeline.__call__.denoising_start) .

<Tip>
يجب أن تكون معلمات `denoising_end` و`denoising_start` عبارة عن رقم عشري بين 0 و1. يتم تمثيل هذه المعلمات كنسبة من خطوات الوقت المتقطعة كما هو محدد بواسطة الجدولة. إذا كنت تستخدم أيضًا معلمة `strength`، فسيتم تجاهلها لأن عدد خطوات إزالة التشويش يتحدد بواسطة خطوات الوقت المتقطعة التي تم تدريب النموذج عليها ونسبة القطع العشرية المعلنة.
</Tip>

دعونا نحدد `denoising_end=0.8` حتى يقوم النموذج الأساسي بأداء أول 80% من إزالة تشويش خطوات **الضوضاء العالية**، ونحدد `denoising_start=0.8` حتى يقوم نموذج التحسين بأداء آخر 20% من إزالة تشويش خطوات **الضوضاء المنخفضة**. يجب أن يكون إخراج النموذج الأساسي في مساحة **الكمون** بدلاً من صورة PIL.

```py
prompt = "أسد مهيب يقفز من حجر كبير في الليل"

image = base(
prompt=prompt،
num_inference_steps=40،
denoising_end=0.8،
output_type="latent"،
).images
image = refiner(
prompt=prompt،
num_inference_steps=40،
denoising_start=0.8،
image=image،
).images[0]
image
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lion_base.png" alt="صورة مُنشأة لأسد على صخرة في الليل" />
<figcaption class="mt-2 text-center text-sm text-gray-500">النموذج الأساسي الافتراضي</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lion_refined.png" alt="صورة مُنشأة لأسد على صخرة في الليل بجودة أعلى" />
<figcaption class="mt-2 text-center text-sm text-gray-500">مجموعة من خبراء إزالة التشويش</figcaption>
</div>
</div>

يمكن أيضًا استخدام نموذج التحسين للتحسين في [`StableDiffusionXLInpaintPipeline`] :

```py
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image, make_image_grid
import torch

base = StableDiffusionXLInpaintPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0"، torch_dtype=torch.float16، variant="fp16"، use_safetensors=True
).to("cuda")

refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-refiner-1.0"،
text_encoder_2=base.text_encoder_2،
vae=base.vae،
torch_dtype=torch.float16،
use_safetensors=True،
variant="fp16"،
).to("cuda")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = load_image(img_url)
mask_image = load_image(mask_url)

prompt = "نمر مهيب يجلس على مقعد"
num_inference_steps = 75
high_noise_frac = 0.7

image = base(
prompt=prompt،
image=init_image،
mask_image=mask_image،
num_inference_steps=num_inference_steps،
denoising_end=high_noise_frac،
output_type="latent"،
).images
image = refiner(
prompt=prompt،
image=image،
mask_image=mask_image،
num_inference_steps=num_inference_steps،
denoising_start=high_noise_frac،
).images[0]
make_image_grid([init_image، mask_image، image.resize((512، 512))], rows=1، cols=3)
```

تعمل طريقة مجموعة الخبراء في إزالة التشويش هذه بشكل جيد لجميع الجداول الزمنية المتاحة!
### من النموذج الأساسي إلى النموذج المحسن:

تحصل SDXL على تعزيز في جودة الصورة من خلال استخدام النموذج المحسن لإضافة تفاصيل عالية الجودة إضافية إلى الصورة الخالية تمامًا من التشويش من النموذج الأساسي، في إعداد الصورة إلى الصورة.

قم بتحميل النماذج الأساسية والمحسنة:

```py
from diffusers import DiffusionPipeline
import torch

base = DiffusionPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

refiner = DiffusionPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-refiner-1.0",
text_encoder_2=base.text_encoder_2,
vae=base.vae,
torch_dtype=torch.float16,
use_safetensors=True,
variant="fp16",
).to("cuda")
```

قم بتوليد صورة من النموذج الأساسي، وقم بتعيين إخراج النموذج إلى مساحة **latent**:

```py
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

image = base(prompt=prompt, output_type="latent").images[0]
```

مرر الصورة المولدة إلى النموذج المحسن:

```py
image = refiner(prompt=prompt, image=image[None, :]).images[0]
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/sd_xl/init_image.png" alt="generated image of an astronaut riding a green horse on Mars" />
<figcaption class="mt-2 text-center text-sm text-gray-500">النموذج الأساسي</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/sd_xl/refined_image.png" alt="higher quality generated image of an astronaut riding a green horse on Mars" />
<figcaption class="mt-2 text-center text-sm text-gray-500">النموذج الأساسي + النموذج المحسن</figcaption>
</div>
</div>

بالنسبة للرسم على الصور، قم بتحميل النموذج الأساسي والنموذج المحسن في [`StableDiffusionXLInpaintPipeline`]، وأزل معلمات `denoising_end` و`denoising_start`، واختر عددًا أقل من خطوات الاستنتاج للنموذج المحسن.

## التكييف الدقيق

ينطوي تدريب SDXL على عدة تقنيات تكييف إضافية، يشار إليها باسم *التكييف الدقيق*. تشمل هذه التقنيات حجم الصورة الأصلي وحجم الصورة المستهدفة ومعلمات الاقتصاص. يمكن استخدام التكييفات الدقيقة في وقت الاستنتاج لإنشاء صور عالية الجودة ومركزة.

<Tip>
يمكنك استخدام كل من معلمات التكييف الدقيق والتكييف الدقيق السلبي بفضل التوجيه الخالي من التصنيف. وهي متاحة في [`StableDiffusionXLPipeline`] و [`StableDiffusionXLImg2ImgPipeline`] و [`StableDiffusionXLInpaintPipeline`] و [`StableDiffusionXLControlNetPipeline`].
</Tip>

### التكييف بالحجم

هناك نوعان من التكييف بالحجم:

- ينشأ تكييف [`original_size`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline.__call__.original_size) من الصور التي تم تغيير حجمها في الدفعة التدريبية (لأنه سيكون من غير المجدي التخلص من الصور الأصغر التي تشكل ما يقرب من 40% من إجمالي بيانات التدريب). بهذه الطريقة، يتعلم SDXL أن آثار تغيير الحجم لا ينبغي أن تكون موجودة في الصور عالية الدقة. أثناء الاستنتاج، يمكنك استخدام `original_size` للإشارة إلى دقة الصورة الأصلية. ينتج عن استخدام القيمة الافتراضية لـ `(1024، 1024)` صور عالية الجودة تشبه صور 1024x1024 في مجموعة البيانات. إذا اخترت استخدام دقة أقل، مثل `(256، 256)`، فسيظل النموذج ينشئ صور 1024x1024، ولكنها ستشبه صور الدقة المنخفضة (أنماط أبسط، ضبابية) في مجموعة البيانات.

- ينشأ تكييف [`target_size`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline.__call__.target_size) من ضبط دقة SDXL لدعم نسب عرض مختلفة للصور. أثناء الاستنتاج، إذا استخدمت القيمة الافتراضية لـ `(1024، 1024)`، فستحصل على صورة تشبه تكوين الصور المربعة في مجموعة البيانات. نوصي باستخدام نفس القيمة لـ `target_size` و`original_size`، ولكن لا تتردد في تجربة الخيارات الأخرى!

يسمح لك 🤗 Diffusers أيضًا بتحديد شروط سلبية حول حجم صورة لتوجيه التوليد بعيدًا عن دقات صور معينة:

```py
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(
prompt=prompt,
negative_original_size=(512, 512),
negative_target_size=(1024, 1024),
).images[0]
```

<div class="flex flex-col justify-center">
<img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/sd_xl/negative_conditions.png"/>
<figcaption class="text-center">الصور المشروطة سلبًا بدقة صور (128، 128)، (256، 256)، و (512، 512).</figcaption>
</div>

### التكييف بالاقتصاص

قد تبدو الصور التي تم إنشاؤها بواسطة نماذج Stable Diffusion السابقة مقصوصة في بعض الأحيان. ويرجع ذلك إلى أن الصور يتم اقتصاصها بالفعل أثناء التدريب بحيث يكون لجميع الصور في دفعة ما نفس الحجم. من خلال التكييف باستخدام إحداثيات الاقتصاص، يتعلم SDXL أن عدم اقتصاص - الإحداثيات `(0، 0)` - يرتبط عادةً بمواضيع مركزية ووجوه كاملة (وهذه هي القيمة الافتراضية في 🤗 Diffusers). يمكنك تجربة إحداثيات مختلفة إذا كنت تريد إنشاء تكوينات غير مركزية!

```py
from diffusers import StableDiffusionXLPipeline
import torch

pipeline = StableDiffusionXLPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline(prompt=prompt, crops_coords_top_left=(256, 0)).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-cropped.png" alt="generated image of an astronaut in a jungle, slightly cropped"/>
</div>

يمكنك أيضًا تحديد إحداثيات الاقتصاص السلبية لتوجيه التوليد بعيدًا عن معلمات الاقتصاص معينة:

```py
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(
prompt=prompt,
negative_original_size=(512, 512),
negative_crops_coords_top_left=(0, 0),
negative_target_size=(1024, 1024),
).images[0]
image
```

## استخدم موجهًا مختلفًا لكل مشفر نصي

يستخدم SDXL مشفرين نصيين، لذا فمن الممكن تمرير موجه مختلف لكل مشفر نصي، والذي يمكن أن [يحسن الجودة](https://github.com/huggingface/diffusers/issues/4004#issuecomment-1627764201). قم بتمرير موجهك الأصلي إلى `prompt` والموجه الثاني إلى `prompt_2` (استخدم `negative_prompt` و`negative_prompt_2` إذا كنت تستخدم موجهات سلبية):

```py
from diffusers import StableDiffusionXLPipeline
import torch

pipeline = StableDiffusionXLPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

# prompt is passed to OAI CLIP-ViT/L-14
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# prompt_2 is passed to OpenCLIP-ViT/bigG-14
prompt_2 = "Van Gogh painting"
image = pipeline(prompt=prompt, prompt_2=prompt_2).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-double-prompt.png" alt="generated image of an astronaut in a jungle in the style of a van gogh painting"/>
</div>

تدعم المشفرات النصية المزدوجة أيضًا تضمينات الانعكاس النصي التي يجب تحميلها بشكل منفصل كما هو موضح في قسم [SDXL textual inversion](textual_inversion_inference#stable-diffusion-xl) .

## التحسينات

SDXL هو نموذج كبير، وقد تحتاج إلى تحسين الذاكرة لجعله يعمل على أجهزتك. فيما يلي بعض النصائح لتوفير الذاكرة وتسريع الاستنتاج.

1. قم بتفريغ النموذج إلى وحدة المعالجة المركزية باستخدام [`~StableDiffusionXLPipeline.enable_model_cpu_offload`] لأخطاء عدم كفاية الذاكرة:

```diff
- base.to("cuda")
- refiner.to("cuda")
+ base.enable_model_cpu_offload()
+ refiner.enable_model_cpu_offload()
```

2. استخدم `torch.compile` للحصول على زيادة في السرعة بنسبة 20% (تحتاج إلى `torch>=2.0`):

```diff
+ base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
+ refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
```

3. قم بتمكين [xFormers](../optimization/xformers) لتشغيل SDXL إذا كان `torch<2.0`:

```diff
+ base.enable_xformers_memory_efficient_attention()
+ refiner.enable_xformers_memory_efficient_attention()
```

## موارد أخرى

إذا كنت مهتمًا بتجربة إصدار بسيط من [`UNet2DConditionModel`] المستخدم في SDXL، فالق نظرة على تنفيذ [minSDXL](https://github.com/cloneofsimo/minSDXL) المكتوب في PyTorch والمتوافق مباشرة مع 🤗 Diffusers.