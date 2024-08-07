
# إكمال الصور

تعد تقنية إكمال الصور (Inpainting) أداة مفيدة لإصلاح الصور عن طريق استبدال أو تعديل مناطق محددة منها. يمكن استخدامها لإزالة العيوب والتشويش، أو حتى استبدال منطقة في الصورة بشيء جديد تمامًا. تعتمد هذه التقنية على قناع لتحديد المناطق المراد ملؤها في الصورة؛ حيث تمثل البكسلات البيضاء المنطقة المراد تعديلها، بينما تمثل البكسلات السوداء المنطقة التي سيتم الاحتفاظ بها دون تغيير. يتم ملء البكسلات البيضاء بناءً على المحتوى المطلوب من خلال النص الوصفي.

مع مكتبة 🤗 Diffusers، يمكنك القيام بإكمال الصور على النحو التالي:

1. قم بتحميل نقطة تفتيش (Checkpoint) خاصة بإكمال الصور باستخدام الفئة [`AutoPipelineForInpainting`]. سيتم الكشف تلقائيًا عن فئة الأنابيب المناسبة لتحميلها بناءً على نقطة التفتيش:

```py
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
"kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()
# احذف السطر التالي إذا لم يكن xFormers مثبتًا أو إذا كان لديك PyTorch 2.0 أو أعلى مثبتًا
pipeline.enable_xformers_memory_efficient_attention()
```

<Tip>
ستلاحظ خلال هذا الدليل أننا نستخدم [`~DiffusionPipeline.enable_model_cpu_offload`] و [`~DiffusionPipeline.enable_xformers_memory_efficient_attention`]، لتوفير الذاكرة وزيادة سرعة الاستدلال. إذا كنت تستخدم PyTorch 2.0، فلا يلزم استدعاء [`~DiffusionPipeline.enable_xformers_memory_efficient_attention`] على خط أنابيبك لأنها ستستخدم بالفعل انتباه [scaled-dot product](../optimization/torch2.0#scaled-dot-product-attention) الأصلي في PyTorch 2.0.
</Tip>

2. قم بتحميل الصورة الأساسية وصورة القناع:

```py
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")
```

3. قم بإنشاء نص وصفي لإكمال الصورة، ثم مرره إلى خط الأنابيب مع الصورة الأساسية وصورة القناع:

```py
prompt = "a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k"
negative_prompt = "bad anatomy, deformed, ugly, disfigured"
image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image).images[0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">base image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">mask image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-cat.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>


## إنشاء صورة القناع

على الرغم من توفير صورة القناع في جميع أمثلة التعليمات البرمجية في هذا الدليل، إلا أنه يمكنك إكمال الصور الخاصة بك، ولكن ستحتاج إلى إنشاء صورة قناع لها. يمكنك استخدام المساحة أدناه لإنشاء صورة قناع بسهولة.

قم بتحميل صورة أساسية لإكمالها، ثم استخدم أداة الرسم لتكوين القناع. بمجرد الانتهاء، انقر فوق "Run" لإنشاء صورة القناع وتحميلها.

<iframe
  src="https://stevhliu-inpaint-mask-maker.hf.space"
  frameborder="0"
  width="850"
  height="450"
></iframe>

## درجة ضبابية القناع

توفر طريقة [`~VaeImageProcessor.blur`] خيارًا لطريقة مزج الصورة الأصلية ومنطقة الإكمال. يتم تحديد مقدار الضبابية بواسطة معامل `blur_factor`. يؤدي زيادة `blur_factor` إلى زيادة مقدار الضبابية المطبقة على حواف القناع، مما يجعل الانتقال بين الصورة الأصلية ومنطقة الإكمال أكثر نعومة. بينما يحافظ معامل `blur_factor` المنخفض أو الصفري على حواف القناع الحادة.

لاستخدام هذه الميزة، قم بإنشاء قناع ضبابي باستخدام معالج الصور:

```py
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from PIL import Image

pipeline = AutoPipelineForInpainting.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to('cuda')

mask = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore_mask.png")
blurred_mask = pipeline.mask_processor.blur(mask, blur_factor=33)
blurred_mask
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore_mask.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">mask with no blur</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/mask_blurred.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">mask with blur applied</figcaption>
  </div>
</div>

## النماذج الشائعة

من بين النماذج الأكثر شعبية لإكمال الصور: [Stable Diffusion Inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)، و [Stable Diffusion XL (SDXL) Inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)، و [Kandinsky 2.2 Inpainting](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint). عادة ما ينتج نموذج SDXL صورًا ذات دقة أعلى من Stable Diffusion v1.5، كما أن Kandinsky 2.2 قادر أيضًا على توليد صور عالية الجودة.

### Stable Diffusion Inpainting

Stable Diffusion Inpainting هو نموذج انتشار لاتيني تمت معايرته على صور بحجم 512x512 لإكمال الصور. إنه نقطة انطلاق جيدة لأنه سريع نسبيًا وينتج صورًا عالية الجودة. لاستخدام هذا النموذج لإكمال الصور، ستحتاج إلى تمرير نص وصفي وصورة أساسية وصورة قناع إلى خط الأنابيب:

```py
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
"runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()
# احذف السطر التالي إذا لم يكن xFormers مثبتًا أو إذا كان لديك PyTorch 2.0 أو أعلى مثبتًا
pipeline.enable_xformers_memory_efficient_attention()

# قم بتحميل الصورة الأساسية وصورة القناع
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

generator = torch.Generator("cuda").manual_seed(92)
prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```

### Stable Diffusion XL (SDXL) Inpainting

SDXL هو إصدار أكبر وأكثر قوة من Stable Diffusion v1.5. يمكن أن يتبع هذا النموذج عملية مكونة من مرحلتين (على الرغم من أنه يمكن استخدام كل نموذج بمفرده)؛ حيث يقوم النموذج الأساسي بتوليد صورة، ثم يقوم نموذج التحسين بأخذ تلك الصورة وتحسين تفاصيلها وجودتها. يمكنك الاطلاع على دليل [SDXL](sdxl) للحصول على دليل أكثر شمولاً حول كيفية استخدام SDXL وتكوين معالمه.

```py
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
"diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()
# احذف السطر التالي إذا لم يكن xFormers مثبتًا أو إذا كان لديك PyTorch 2.0 أو أعلى مثبتًا
pipeline.enable_xformers_memory_efficient_attention()

# قم بتحميل الصورة الأساسية وصورة القناع
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_Intersecting image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

generator = torch.Generator("cuda").manual_seed(92)
prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```

### Kandinsky 2.2 Inpainting

تتشابه عائلة نماذج Kandinsky مع SDXL لأنها تستخدم نموذجين أيضًا؛ حيث يقوم نموذج الصورة الأولية بإنشاء تضمينات الصورة، وينشئ نموذج الانتشار الصور منها. يمكنك تحميل صورة أولية ونموذج انتشار بشكل منفصل، ولكن أسهل طريقة لاستخدام Kandinsky 2.2 هي تحميله في فئة [`AutoPipelineForInpainting`] التي تستخدم [`KandinskyV22InpaintCombinedPipeline`] تحت الغطاء.

يمكنك العثور على النص البرمجي الكامل أدناه. لاحظ أنه يستخدم مكتبة xFormers، لذا تأكد من تثبيتها قبل تشغيله.

تُظهر الصور أدناه نتائج استخدام Kandinsky 2.2 لإكمال الصورة الناقصة. لاحظ كيف أن النتيجة أكثر تفصيلاً ووضوحًا من SD وSDXL.

```py
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

generator = torch.Generator("cuda").manual_seed(92)
prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```


<div class="flex flex-row gap-4">
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأساسية</figcaption>
</div>
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-sdv1.5.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">Stable Diffusion Inpainting</figcaption>
</div>
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-sdxl.png"/>
<figcaption class="mt-turut text-center text-sm text-gray-500">Stable Diffusion XL Inpainting</figcaption>
</div>
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-kandinsky.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">Kandinsky 2.2 Inpainting</figcaption>
</div>
</div>


## نقاط تفتيش غير خاصة بـ Inpaint

حتى الآن، استخدم هذا الدليل نقاط تفتيش خاصة بـ Inpaint مثل [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting). ولكن يمكنك أيضًا استخدام نقاط تفتيش عادية مثل [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5). دعونا نقارن نتائج نقطتي التفتيش.

الصورة على اليسار مُولدة من نقطة تفتيش عادية، والصورة على اليمين من نقطة تفتيش Inpaint. ستلاحظ على الفور أن الصورة على اليسار ليست بنفس الجودة، ويمكنك رؤية مخطط المنطقة التي يُفترض أن تقوم بها النموذج Inpaint. الصورة على اليمين أنظف بكثير، وتظهر المنطقة التي تم إصلاحها بشكل أكثر طبيعية.

<hfoptions id="regular-specific">
<hfoption id="runwayml/stable-diffusion-v1-5">

```py
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

generator = torch.Generator("cuda").manual_seed(92)
prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

</hfoption>
<hfoption id="runwayml/stable-diffusion-inpainting">

```py
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

generator = torch.Generator("cuda").manual_seed(92)
prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

</hfoption>
</hfoptions>

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/non-inpaint-specific.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">runwayml/stable-diffusion-v1-5</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-specific.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">runwayml/stable-diffusion-inpainting</figcaption>
  </div>
</div>


على الرغم من ذلك، بالنسبة للمهام الأساسية مثل إزالة كائن من صورة (مثل الصخور على الطريق، على سبيل المثال)، فإن نقطة تفتيش عادية تعطي نتائج جيدة جدًا. ليس هناك فرق ملحوظ بين نقطة التفتيش العادية ونقطة تفتيش Inpaint.

<hfoptions id="inpaint">
<hfoption id="runwayml/stable-diffusion-v1-5">

```py
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/road-mask.png")

image = pipeline(prompt="road", image=init_image, mask_image=mask_image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

</hfoption>
<hfoption id="runwayml/stable-diffusion-inpaint">

```py
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/road-mask.png")

image = pipeline(prompt="road", image=init_image, mask_image=mask_image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

</hfoption>
</hfoptions>

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/regular-inpaint-basic.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">runwayml/stable-diffusion-v1-5</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/specific-inpaint-basic.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">runwayml/stable-diffusion-inpainting</figcaption>
  </div>
</div>


المقايضة لاستخدام نقطة تفتيش غير خاصة بـ Inpaint هي أن جودة الصورة الإجمالية قد تكون أقل، ولكنها تميل عمومًا إلى الحفاظ على منطقة القناع (وهذا هو السبب في إمكانية رؤية مخطط القناع). يتم تدريب نقاط تفتيش Inpaint الخاصة عن قصد لإنشاء صور ذات جودة أعلى، والتي تتضمن إنشاء انتقال أكثر طبيعية بين المناطق المقنعة وغير المقنعة. ونتيجة لذلك، من المرجح أن تغير هذه النقاط منطقة غير المقنعة.

إذا كان الحفاظ على المنطقة غير المقنعة أمرًا مهمًا لمهمتك، فيمكنك استخدام طريقة [`VaeImageProcessor.apply_overlay`] لإجبار المنطقة غير المقنعة للصورة على البقاء كما هي على حساب بعض الانتقالات غير الطبيعية بين المناطق المقنعة وغير المقنعة.


```py
import PIL
import numpy as np
import torch

from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

device = "cuda"
pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
)
pipeline = pipeline.to(device)

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = load_image(img_url).resize((512, 512))
mask_image = load_image(mask_url).resize((512, 512))

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
repainted_image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
repainted_image.save("repainted_image.png")

unmasked_unchanged_image = pipeline.image_processor.apply_overlay(mask_image, init_image, repainted_image)
unmasked_unchanged_image.save("force_unmasked_unchanged.png")
make_image_grid([init_image, mask_image, repainted_image, unmasked_unchanged_image], rows=2, cols=2)
```



## تكوين معلمات الأنابيب

تعتمد ميزات الصورة - مثل الجودة و"الإبداع" - على معلمات الأنابيب. من المهم معرفة ما تفعله هذه المعلمات للحصول على النتائج المرجوة. دعونا نلقي نظرة على أهم المعلمات ونرى كيف يؤثر تغييرها على الإخراج.

### القوة

`strength` هي مقياس لكمية الضوضاء المضافة إلى الصورة الأساسية، والتي تؤثر على مدى تشابه الإخراج مع الصورة الأساسية.

- 📈 قيمة عالية لـ `strength` تعني إضافة المزيد من الضوضاء إلى الصورة وتستغرق عملية إزالة التشويش وقتًا أطول، ولكنك ستحصل على صور عالية الجودة تختلف أكثر عن الصورة الأساسية

- 📉 قيمة منخفضة لـ `strength` تعني إضافة قدر أقل من الضوضاء إلى الصورة وتكون عملية إزالة التشويش أسرع، ولكن قد لا تكون جودة الصورة جيدة وقد تشبه الصورة المولدة الصورة الأساسية أكثر


```py
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, strength=0.6).images[0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-strength-0.6.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">strength = 0.6</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-strength-0.8.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">strength = 0.8</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-strength-1.0.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">strength = 1.0</figcaption>
  </div>
</div>


### مقياس التوجيه

`guidance_scale` يؤثر على مدى توافق موجه النص والصورة المولدة.

- 📈 قيمة عالية لـ `guidance_scale` تعني أن الموجه والصورة المولدة متوافقان بشكل وثيق، لذا فإن الإخراج هو تفسير أكثر صرامة للموجه

- 📉 قيمة منخفضة لـ `guidance_scale` تعني أن الموجه والصورة المولدة متوافقان بشكل فضفاض، لذا فقد يكون الإخراج أكثر تنوعًا عن الموجه

يمكنك استخدام `strength` و`guidance_scale` معًا لمزيد من التحكم في مدى تعبيرية النموذج. على سبيل المثال، يمنح مزيج من القيم العالية لـ `strength` و`guidance_scale` النموذج أكبر قدر من الحرية الإبداعية.

```py
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, guidance_scale=2.5).images[0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-guidance-2.5.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale = 2.5</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-guidance-7.5.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale = 7.5</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-guidance-12.5.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale = 12.5</figcaption>
  </div>
</div>


### موجه سلبي

يفترض الموجه السلبي الدور المعاكس للموجه؛ فهو يوجه النموذج بعيدًا عن توليد أشياء معينة في صورة. وهذا مفيد لتحسين جودة الصورة بسرعة ومنع النموذج من توليد أشياء لا تريدها.

```py
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
negative_prompt = "bad architecture, unstable, poor details, blurry"
image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image).images[0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```

<div class="flex justify-center">
  <figure>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-negative.png" />
    <figcaption class="text-center">negative_prompt = "bad architecture, unstable, poor details, blurry"</figcaption>
  </figure>
</div>


### قناع الحشو المحصول

تتمثل إحدى طرق زيادة جودة صورة الحشو في استخدام معلمة [`padding_mask_crop`](https://huggingface.co/docs/diffusers/v0.25.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline.__call__.padding_mask_crop). عندما يتم تمكين هذا الخيار، فإنه يقوم باقتصاص المنطقة المقنعة ببعض الحشو الذي يحدده المستخدم، كما سيقوم باقتصاص نفس المنطقة من الصورة الأصلية. يتم تكبير كل من الصورة والقناع إلى دقة أعلى للحشو، ثم يتم وضعهما فوق الصورة الأصلية. هذه طريقة سريعة وسهلة لتحسين جودة الصورة دون استخدام خط أنابيب منفصل مثل [`StableDiffusionUpscalePipeline`].

أضف معلمة `padding_mask_crop` إلى مكالمة خط الأنابيب وقم بتعيينها على قيمة الحشو المرغوبة.
```py
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from PIL import Image

generator = torch.Generator(device='cuda').manual_seed(0)
pipeline = AutoPipelineForInpainting.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to('cuda')

base = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png")
mask = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore_mask.png")

image = pipeline("boat", image=base, mask_image=mask, strength=0.75, generator=generator, padding_mask_crop=32).images[0]
image
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/baseline_inpaint.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">default inpaint image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/padding_mask_crop_inpaint.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">inpaint image with `padding_mask_crop` enabled</figcaption>
  </div>
</div>


### خطوط أنابيب الحشو المتسلسلة

يمكن تسلسل [`AutoPipelineForInpainting`] مع خطوط أنابيب أخرى من 🤗 Diffusers لتحرير إخراجها. غالبًا ما يكون هذا مفيدًا لتحسين جودة الإخراج من خطوط أنابيب الانتشار الأخرى، وإذا كنت تستخدم خطوط أنابيب متعددة، فقد يكون من الأكثر كفاءة في الذاكرة تسلسلها معًا للحفاظ على الإخراج في مساحة خفية وإعادة استخدام نفس مكونات خط الأنابيب.

### ربط النص بالصورة ثم إصلاحها

يتيح ربط خط أنابيب من النص إلى الصورة وإصلاحها إمكانية إصلاح الصورة المولدة، ولا يلزمك توفير صورة أساسية للبدء. يجعل هذا الأمر مريحًا لتحرير مخرجات النص إلى الصورة المفضلة لديك دون الحاجة إلى إنشاء صورة جديدة تمامًا.

ابدأ بخط أنابيب من النص إلى الصورة لإنشاء قلعة:


```py
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

text2image = pipeline("concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k").images[0]
```


قم بتحميل صورة القناع من الإخراج أعلاه:
```py
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_text-chain-mask.png")
```

والآن دعونا نصلح المنطقة المقنعة بشلال:

```py
pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

prompt = "digital painting of a fantasy waterfall, cloudy"
image = pipeline(prompt=prompt, image=text2image, mask_image=mask_image).images[0]
make_image_grid([text2image, mask_image, image], rows=1, cols=3)
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-text-chain.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">text-to-image</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-text-chain-out.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">inpaint</figcaption>
  </div>
</div>

### إصلاح إلى صورة إلى صورة

يمكنك أيضًا ربط خط أنابيب الإصلاح قبل خط أنابيب آخر مثل الصورة إلى الصورة أو أداة تكبير لتحسين الجودة.

ابدأ بإصلاح صورة:

```py
import torch
from diffusers import AutoPipelineForInpainting, AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image_inpainting = pipeline(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

# resize image to 1024x1024 for SDXL
image_inpainting = image_inpainting.resize((1024, 1024))
```

الآن دعونا نمرر الصورة إلى خط أنابيب إصلاح آخر مع نموذج SDXL refiner لتحسين تفاصيل الصورة والجودة:

```py
pipeline = AutoPipelineForInpainting.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

image = pipeline(prompt=prompt, image=image_inpainting, mask_image=mask_image, output_type="latent").images[0]
```

<Tip>
من المهم تحديد output_type="latent" في خط الأنابيب للحفاظ على جميع الإخراج في مساحة الكامنة لتجنب خطوة الترميز فك الترميز غير الضرورية. يعمل هذا فقط إذا كانت خطوط الأنابيب المتسلسلة تستخدم نفس VAE. على سبيل المثال، في قسم [النص إلى الصورة إلى الإصلاح](#text-to-image-to-inpaint)، يستخدم Kandinsky 2.2 فئة VAE مختلفة عن نموذج Stable Diffusion لذلك لن يعمل. ولكن إذا استخدمت Stable Diffusion v1.5 لكل من خطوط الأنابيب، فيمكنك الحفاظ على كل شيء في مساحة الكامنة لأنها تستخدم جميعها [`AutoencoderKL`].
</Tip>

أخيرًا، يمكنك تمرير هذه الصورة إلى خط أنابيب الصورة إلى الصورة لوضع اللمسات الأخيرة عليها. من الأكثر كفاءة استخدام طريقة [`~AutoPipelineForImage2Image.from_pipe`] لإعادة استخدام مكونات خط الأنابيب الموجودة، وتجنب تحميل جميع مكونات خط الأنابيب في الذاكرة مرة أخرى دون داع.

```py
pipeline = AutoPipelineForImage2Image.from_pipe(pipeline)
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

image = pipeline(prompt=prompt, image=image).images[0]
make_image_grid([init_image, mask_image, image_inpainting, image], rows=2, cols=2)
```

<div class="flex flex-row gap-4">
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأولية</figcaption>
</div>
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-to-image-chain.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">إصلاح</figcaption>
</div>
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-to-image-final.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة إلى الصورة</figcaption>
</div>
</div>

في الواقع، الإصلاح والصورة إلى الصورة مهام متشابهة جدًا. تقوم الصورة إلى الصورة بتوليد صورة جديدة تشبه الصورة المقدمة الموجودة. يفعل الإصلاح الشيء نفسه، ولكنه يحول فقط منطقة الصورة التي حددها القناع ويظل باقي الصورة دون تغيير. يمكنك اعتبار الإصلاح كأداة أكثر دقة لإجراء تغييرات محددة وللصورة إلى الصورة نطاق أوسع لإجراء تغييرات واسعة النطاق.


## التحكم في إنشاء الصور

من الصعب جعل الصورة تبدو بالضبط كما تريد لأن عملية إزالة التشويش عشوائية. في حين أنه يمكنك التحكم في جوانب معينة من التوليد عن طريق تكوين معلمات مثل "negative_prompt"، هناك طرق أفضل وأكثر كفاءة للتحكم في إنشاء الصور.

### وزن الفحص

يوفر وزن الفحص طريقة قابلة للقياس الكمي لقياس حجم تمثيل المفاهيم في فحص. يمكنك استخدامه لزيادة أو تقليل حجم متجه تضمين النص لكل مفهوم في الفحص، والذي يحدد بعد ذلك مقدار كل مفهوم يتم توليده. توفر مكتبة [Compel](https://github.com/damian0815/compel) بناء جملة بديهي لقياس أوزان الفحص وتوليد التضمينات. تعرف على كيفية إنشاء التضمينات في دليل [وزن الفحص](../using-diffusers/weighted_prompts).

بمجرد إنشاء التضمينات، قم بتمريرها إلى معلمة "prompt_embeds" (و"negative_prompt_embeds" إذا كنت تستخدم فحصًا سلبيًا) في ["AutoPipelineForInpainting"]. تحل التضمينات محل معلمة "الفحص":

```py
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16,
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

image = pipeline(prompt_embeds=prompt_embeds, # generated from Compel
    negative_prompt_embeds=negative_prompt_embeds, # generated from Compel
    image=init_image,
    mask_image=mask_image
).images[0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```



### ControlNet

تُستخدم نماذج ControlNet مع نماذج الانتشار الأخرى مثل Stable Diffusion، وتوفر طريقة أكثر مرونة ودقة للتحكم في كيفية إنشاء الصورة. تقبل ControlNet إدخال صورة تكييف إضافية توجه نموذج الانتشار للحفاظ على الميزات الموجودة فيها.

على سبيل المثال، دعنا نُكيّف صورة باستخدام ControlNet مُدربة مسبقًا على صور Inpaint:

```py
import torch
import numpy as np
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
from diffusers.utils import load_image, make_image_grid

# load ControlNet
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, variant="fp16")

# pass ControlNet to the pipeline
pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

# prepare control image
def make_inpaint_condition(init_image, mask_image):
    init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
    mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0

    assert init_image.shape[0:1] == mask_image.shape[0:1], "image and image_mask must have the same image size"
    init_image[mask_image > 0.5] = -1.0  # set as masked pixel
    init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
    init_image = torch.from_numpy(init_image)
    return init_image

control_image = make_inpaint_condition(init_image, mask_image)
```

الآن، قم بتوليد صورة من الصورة الأساسية وصورة القناع وصورة التحكم. ستلاحظ أن ميزات الصورة الأساسية محفوظة بشدة في الصورة المولدة.

```py
prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, control_image=control_image).images[0]
make_image_grid([init_image, mask_image, PIL.Image.fromarray(np.uint8(control_image[0][0])).convert('RGB'), image], rows=2, cols=2)
```

يمكنك المضي قدمًا خطوة أخرى وسلسلة مع خط أنابيب الصورة إلى الصورة لتطبيق أسلوب جديد:
```py
from diffusers import AutoPipelineForImage2Image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "nitrosocke/elden-ring-diffusion", torch_dtype=torch.float16,
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

prompt = "elden ring style castle" # include the token "elden ring style" in the prompt
negative_prompt = "bad architecture, deformed, disfigured, poor details"

image_elden_ring = pipeline(prompt, negative_prompt=negative_prompt, image=image).images[0]
make_image_grid([init_image, mask_image, image, image_elden_ring], rows=2, cols=2)
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">initial image</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-controlnet.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">ControlNet inpaint</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-img2img.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">image-to-image</figcaption>
  </div>
</div>

## التحسين

قد يكون من الصعب والبطيء تشغيل نماذج الانتشار إذا كنت تعاني من قيود الموارد، ولكن يمكنك استخدام بعض الحيل التحسينية. أحد أكبر التحسينات (وأسهلها) التي يمكنك تمكينها هو التبديل إلى الانتباه الكفء للذاكرة. إذا كنت تستخدم PyTorch 2.0، يتم تمكين الانتباه لنقاط المنتج المُمَيز تلقائيًا، ولا يلزم القيام بأي شيء آخر. بالنسبة لمستخدمي PyTorch غير 2.0، يمكنك تثبيت واستخدام تنفيذ xFormers للاهتمام الكفء للذاكرة. كلا الخيارين يقللان من استخدام الذاكرة ويُسرعان الاستدلال.

يمكنك أيضًا نقل النموذج إلى وحدة المعالجة المركزية لتوفير المزيد من الذاكرة:

```diff
+ pipeline.enable_xformers_memory_efficient_attention()
+ pipeline.enable_model_cpu_offload()
```


لزيادة تسريع رمز الاستدلال الخاص بك، استخدم [torch_compile](../optimization/torch2.0#torchcompile). يجب عليك لف [torch.compile](https://pytorch.org/docs/stable/generated/torch.compile.html#torch.compile) حول المكون الأكثر كثافة في الاستخدام في خط الأنابيب والذي يكون عادةً UNet:

```py
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
```

تعرف على المزيد في أدلة [تقليل استخدام الذاكرة](../optimization/memory) و [Torch 2.0](../optimization/torch2.0).