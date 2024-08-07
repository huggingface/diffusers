# الصورة إلى الصورة
[[open-in-colab]]

الصورة إلى الصورة مشابهة لـ [النص إلى الصورة](conditional_image_generation)، ولكن بالإضافة إلى موجه، يمكنك أيضًا تمرير صورة أولية كنقطة بداية لعملية الانتشار. يتم ترميز الصورة الأولية إلى مساحة خفية ويتم إضافة الضوضاء إليها. ثم تأخذ النموذج الخفي للانتشار موجه والصورة الخفية الصاخبة، ويتوقع الضوضاء المضافة، ويزيل الضوضاء المتوقعة من الصورة الأولية الخفية للحصول على الصورة الخفية الجديدة. أخيرًا، يقوم فك الترميز بترجمة الصورة الخفية الجديدة مرة أخرى إلى صورة.

مع 🤗 Diffusers، هذا سهل مثل 1-2-3:

1. قم بتحميل نقطة تفتيش في فئة [`AutoPipelineForImage2Image`]؛ يقوم هذا الأنبوب تلقائيًا بتعيين تحميل فئة الأنابيب الصحيحة بناءً على نقطة التفتيش:

```py
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()
```

<Tip>
ستلاحظ طوال الدليل، نستخدم [`~DiffusionPipeline.enable_model_cpu_offload`] و [`~DiffusionPipeline.enable_xformers_memory_efficient_attention`]، لتوفير الذاكرة وزيادة سرعة الاستدلال. إذا كنت تستخدم PyTorch 2.0، فلست بحاجة إلى استدعاء [`~DiffusionPipeline.enable_xformers_memory_efficient_attention`] على خط أنابيبك لأنه سيكون بالفعل باستخدام اهتمام PyTorch 2.0 الأصلي [scaled-dot product](../optimization/torch2.0#scaled-dot-product-attention).
</Tip>

2. قم بتحميل صورة لتمريرها إلى خط الأنابيب:

```py
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
```

3. قم بتمرير موجه وصورة إلى خط الأنابيب لإنشاء صورة:

```py
prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
image = pipeline(prompt, image=init_image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500>الصورة الأولية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500>الصورة المولدة</figcaption>
</div>
</div>

## النماذج الشعبية

أكثر نماذج الصور إلى الصور شيوعًا هي [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)، [Stable Diffusion XL (SDXL)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)، و [Kandinsky 2.2](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder). تختلف نتائج نماذج Stable Diffusion و Kandinsky بسبب الاختلافات في بنيتها وعملية تدريبها؛ يمكنك عمومًا توقع أن تنتج SDXL صورًا ذات جودة أعلى من Stable Diffusion v1.5. دعنا نلقي نظرة سريعة على كيفية استخدام كل من هذه النماذج ومقارنة نتائجها.

### الانتشار المستقر v1.5

Stable Diffusion v1.5 عبارة عن نموذج انتشار خفي يتم تهيئته من نقطة تفتيش مبكرة، ويتم ضبط دقته بشكل أكبر لـ 595 ألف خطوة على صور 512x512. لاستخدام هذا الأنبوب للصورة إلى الصورة، ستحتاج إلى إعداد صورة أولية لتمريرها إلى خط الأنابيب. ثم يمكنك تمرير موجه والصورة إلى خط الأنابيب لإنشاء صورة جديدة:

```py
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500>الصورة الأولية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdv1.5.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500>الصورة المولدة</figcaption>
</div>
</div>

### الانتشار المستقر XL (SDXL)

SDXL هو إصدار أكثر قوة من نموذج Stable Diffusion. يستخدم نموذجًا أساسيًا أكبر، ونموذجًا محسنًا إضافيًا لزيادة جودة إخراج النموذج الأساسي. اقرأ دليل [SDXL](sdxl) للحصول على دليل أكثر تفصيلاً حول كيفية استخدام هذا النموذج، والتقنيات الأخرى التي يستخدمها لإنتاج صور عالية الجودة.

```py
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdxl-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image, strength=0.5).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdxl-init.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500>الصورة الأولية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdxl.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500>الصورة المولدة</figcaption>
</div>
</div>

### كاندينسكي 2.2

يختلف نموذج كاندينسكي عن نماذج Stable Diffusion لأنه يستخدم نموذجًا مسبقًا للصور لإنشاء تضمينات الصور. تساعد التضمينات في إنشاء محاذاة أفضل بين النص والصور، مما يسمح لنموذج الانتشار الخفي بتوليد صور أفضل.

أبسط طريقة لاستخدام Kandinsky 2.2 هي:

```py
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500>الصورة الأولية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-kandinsky.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500>الصورة المولدة</figcaption>
</div>
</div>

## تكوين معلمات خط الأنابيب

هناك العديد من المعلمات المهمة التي يمكنك تكوينها في خط الأنابيب والتي ستؤثر على عملية إنشاء الصور وجودة الصورة. دعنا نلقي نظرة فاحصة على ما تفعله هذه المعلمات وكيف يؤثر تغييرها على الإخراج.
### القوة

`strength` هي أحد أهم المعلمات التي يجب مراعاتها، وسيكون لها تأثير كبير على الصورة التي يتم إنشاؤها. فهي تحدد مدى تشابه الصورة الناتجة مع الصورة الأولية. وبعبارة أخرى:

- 📈 تعطي قيمة أعلى لـ `strength` النموذج مزيدًا من "الإبداع" لإنشاء صورة مختلفة عن الصورة الأولية؛ وتعني قيمة `strength` تساوي 1.0 تجاهل الصورة الأولية إلى حد كبير

- 📉 تعني قيمة أقل لـ `strength` أن الصورة الناتجة أكثر تشابها مع الصورة الأولية

يرتبط معلم `strength` و`num_inference_steps` لأن `strength` تحدد عدد خطوات الضجيج المراد إضافتها. على سبيل المثال، إذا كان `num_inference_steps` هو 50 و`strength` هو 0.8، فهذا يعني إضافة 40 (50 * 0.8) خطوة ضجيج إلى الصورة الأولية ثم إزالة الضجيج لـ 40 خطوة للحصول على الصورة المولدة حديثًا.

```py
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image, strength=0.8).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div class="flex flex-row gap-4">
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-strength-0.4.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">strength = 0.4</figcaption>
</div>
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-strength-0.6.png"/>
<figcaption class="mt-₂ text-center text-sm text-gray-500">strength = 0.6</figcaption>
</div>
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-strength-1.0.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">strength = 1.0</figcaption>
</div>
</div>

### مقياس التوجيه

يُستخدم معلم `guidance_scale` للتحكم في مدى توافق الصورة المولدة وطلب النص. تعني قيمة أعلى لـ `guidance_scale` أن الصورة المولدة أكثر توافقا مع الطلب، في حين أن قيمة أقل لـ `guidance_scale` تعني أن الصورة المولدة لديها مساحة أكبر للانحراف عن الطلب.

يمكنك الجمع بين `guidance_scale` و`strength` لمزيد من التحكم الدقيق في مدى تعبير النموذج. على سبيل المثال، قم بدمج `strength + guidance_scale` عاليًا للإبداع الأقصى أو استخدم مزيجًا من `strength` المنخفض و`guidance_scale` المنخفض لإنشاء صورة تشبه الصورة الأولية ولكنها ليست مرتبطة بشكل صارم بالطلب.

```py
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image, guidance_scale=8.0).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div class="flex flex-row gap-4">
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-guidance-0.1.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale = 0.1</figcaption>
</div>
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-guidance-3.0.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale = 5.0</figcaption>
</div>
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-guidance-7.5.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale = 10.0</figcaption>
</div>
</div>

### طلب سلبي

يؤدي الطلب السلبي إلى تهيئة النموذج لعدم تضمين أشياء في صورة، ويمكن استخدامه لتحسين جودة الصورة أو تعديلها. على سبيل المثال، يمكنك تحسين جودة الصورة عن طريق تضمين مطالبات سلبية مثل "تفاصيل سيئة" أو "ضبابي" لتشجيع النموذج على إنشاء صورة ذات جودة أعلى. أو يمكنك تعديل صورة عن طريق تحديد الأشياء التي سيتم استبعادها من الصورة.

```py
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
"stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
negative_prompt = "ugly, deformed, disfigured, poor details, bad anatomy"

# pass prompt and image to pipeline
image = pipeline(prompt, negative_prompt=negative_prompt, image=init_image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div class="flex flex-row gap-4">
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-negative-1.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">negative_prompt = "ugly, deformed, disfigured, poor details, bad anatomy"</figcaption>
</div>
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-negative-2.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">negative_prompt = "jungle"</figcaption>
</div>
</div>

## خطوط أنابيب الصورة إلى الصورة المتسلسلة

هناك بعض الطرق المثيرة للاهتمام الأخرى التي يمكنك من خلالها استخدام خط أنابيب الصورة إلى الصورة بخلاف مجرد إنشاء صورة (على الرغم من أن هذا رائع أيضًا). يمكنك المضي قدمًا وربطه بخطوط أنابيب أخرى.

### النص إلى الصورة إلى الصورة

يسمح ربط خط أنابيب النص إلى الصورة والصورة إلى الصورة بإنشاء صورة من النص واستخدام الصورة المولدة كصورة أولية لخط أنابيب الصورة إلى الصورة. هذا مفيد إذا كنت تريد إنشاء صورة من الصفر. على سبيل المثال، دعنا نقوم بتسلسل نموذج Stable Diffusion ونموذج Kandinsky.

ابدأ بإنشاء صورة باستخدام خط أنابيب النص إلى الصورة:

```py
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch
from diffusers.utils import make_image_grid

pipeline = AutoPipelineForText2Image.from_pretrained(
"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

text2image = pipeline("Astronaut in a jungle, cold color palette, muted colors, detailed, 8k").images[0]
text2image
```

الآن يمكنك تمرير هذه الصورة المولدة إلى خط أنابيب الصورة إلى الصورة:

```py
pipeline = AutoPipelineForImage2Image.from_pretrained(
"kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

image2image = pipeline("Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", image=text2image).images[0]
make_image_grid([text2image, image2image], rows=1, cols=2)
```

### ربط عدة أنابيب للصور مع بعضها البعض
يمكنك أيضًا ربط عدة أنابيب للصور مع بعضها البعض لإنشاء صور أكثر إثارة للاهتمام. يمكن أن يكون هذا مفيدًا لأداء نقل الأسلوب بشكل تكراري على صورة، أو إنشاء صور GIF قصيرة، أو استعادة الألوان لصورة، أو استعادة المناطق المفقودة من صورة.

ابدأ بتوليد صورة:

```py
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image, output_type="latent").images[0]
```

<Tip>
من المهم تحديد `output_type="latent"` في الأنبوب للحفاظ على جميع المخرجات في المساحة المخفية لتجنب خطوة الترميز والفك غير الضرورية. يعمل هذا فقط إذا كانت الأنابيب المرتبطة تستخدم نفس VAE.
</Tip>

مرر الإخراج المخفي من هذا الأنبوب إلى الأنبوب التالي لتوليد صورة على [نمط كتاب هزلي](https://huggingface.co/ogkalu/Comic-Diffusion):

```py
pipeline = AutoPipelineForImage2Image.from_pretrained(
"ogkalu/Comic-Diffusion"، torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# يجب تضمين الرمز "charliebo artstyle" في المطالبة لاستخدام نقطة التفتيش هذه
image = pipeline("Astronaut in a jungle, charliebo artstyle"، image=image, output_type="latent").images[0]
```

كرر مرة أخرى لتوليد الصورة النهائية على [نمط فن البكسل](https://huggingface.co/kohbanye/pixel-art-style):

```py
pipeline = AutoPipelineForImage2Image.from_pretrained(
"kohbanye/pixel-art-style"، torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# يجب تضمين الرمز "pixelartstyle" في المطالبة لاستخدام نقطة التفتيش هذه
image = pipeline("Astronaut in a jungle, pixelartstyle"، image=image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

### أنبوب الصور إلى أنبوب تكبير إلى صورة فائقة الدقة
هناك طريقة أخرى لربط أنبوب الصور الخاص بك وهي استخدام أنبوب تكبير وصورة فائقة الدقة لزيادة مستوى التفاصيل في الصورة حقًا.

ابدأ بأنبوب صورة إلى صورة:

```py
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
"runwayml/stable-diffusion-v1-5"، torch_dtype=torch.float16، variant="fp16"، use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image_1 = pipeline(prompt, image=init_image, output_type="latent").images[0]
```

<Tip>
من المهم تحديد `output_type="latent"` في الأنبوب للحفاظ على جميع المخرجات في المساحة المخفية لتجنب خطوة الترميز والفك غير الضرورية. يعمل هذا فقط إذا كانت الأنابيب المرتبطة تستخدم نفس VAE.
</Tip>

قم بتوصيله بأنبوب تكبير لزيادة دقة الصورة:

```py
from diffusers import StableDiffusionLatentUpscalePipeline

upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
"stabilityai/sd-x2-latent-upscaler"، torch_dtype=torch.float16، variant="fp16"، use_safetensors=True
)
upscaler.enable_model_cpu_offload()
upscaler.enable_xformers_memory_efficient_attention()

image_2 = upscaler(prompt، image=image_1، output_type="latent").images[0]
```

أخيرًا، قم بتوصيله بأنبوب فائق الدقة لزيادة تحسين الدقة:

```py
from diffusers import StableDiffusionUpscalePipeline

super_res = StableDiffusionUpscalePipeline.from_pretrained(
"stabilityai/stable-diffusion-x4-upscaler"، torch_dtype=torch.float16، variant="fp16"، use_safetensors=True
)
super_res.enable_model_cpu_offload()
super_res.enable_xformers_memory_efficient_attention()

image_3 = super_res(prompt، image=image_2).images[0]
make_image_grid([init_image, image_3.resize((512, 512))], rows=1, cols=2)
```

## التحكم في إنشاء الصور
يمكن أن يكون محاولة إنشاء صورة تبدو بالضبط كما تريد أمرًا صعبًا، وهذا هو السبب في أن تقنيات ونماذج التحكم في التوليد مفيدة جدًا. في حين يمكنك استخدام `negative_prompt` للتحكم جزئيًا في إنشاء الصور، هناك طرق أكثر قوة مثل وزن المطالبة وشبكات التحكم.

### وزن المطالبة
يسمح وزن المطالبة بتغيير حجم تمثيل كل مفهوم في مطالبة. على سبيل المثال، في مطالبة مثل "رائد فضاء في الغابة، لوحة ألوان باردة، ألوان خافتة، مفصلة، 8k"، يمكنك اختيار زيادة أو تقليل تضمين "رائد الفضاء" و"الغابة". توفر مكتبة [Compel](https://github.com/damian0815/compel) بناء جملة بسيطًا لتعديل أوزان المطالبة وإنشاء التضمينات. يمكنك معرفة كيفية إنشاء التضمينات في دليل [وزن المطالبة](weighted_prompts).

لدى [`AutoPipelineForImage2Image`] معلمة `prompt_embeds` (و`negative_prompt_embeds` إذا كنت تستخدم مطالبة سلبية) حيث يمكنك تمرير التضمينات التي تحل محل معلمة `prompt`.

```py
from diffusers import AutoPipelineForImage2Image
import torch

pipeline = AutoPipelineForImage2Image.from_pretrained(
"runwayml/stable-diffusion-v1-5"، torch_dtype=torch.float16، variant="fp16"، use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

image = pipeline(prompt_embeds=prompt_embeds، # تم إنشاؤه من Compel
negative_prompt_embeds=negative_prompt_embeds، # تم إنشاؤه من Compel
image=init_image،
).images[0]
```

### ControlNet

توفر ControlNets طريقة أكثر مرونة ودقة للتحكم في توليد الصور لأنك يمكن أن تستخدم صورة شرطية إضافية. يمكن أن تكون الصورة الشرطية صورة Canny أو خريطة عمق أو تجزئة صورة، وحتى الخربشات! أيًا كان نوع الصورة الشرطية التي تختارها، يقوم ControlNet بتوليد صورة تحافظ على المعلومات الموجودة فيها.

على سبيل المثال، دعونا نشترط صورة باستخدام خريطة عمق للحفاظ على المعلومات المكانية في الصورة.

```py
from diffusers.utils import load_image, make_image_grid

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)
init_image = init_image.resize((958, 960)) # تغيير حجم الصورة إلى أبعاد صورة العمق
depth_image = load_image("https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth/resolve/main/images/control.png")
make_image_grid([init_image, depth_image], rows=1, cols=2)
```

قم بتحميل نموذج ControlNet المشروط بخرائط العمق و [`AutoPipelineForImage2Image`]:

```py
from diffusers import ControlNetModel, AutoPipelineForImage2Image
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipeline = AutoPipelineForImage2Image.from_pretrained(
"runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# احذف السطر التالي إذا لم يكن xFormers مثبتًا أو إذا كان لديك PyTorch 2.0 أو أعلى مثبتًا
pipeline.enable_xformers_memory_efficient_attention()
```

الآن قم بتوليد صورة جديدة مشروطة بخريطة العمق والصورة الأولية والنص الفعلي:

```py
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image_control_net = pipeline(prompt, image=init_image, control_image=depth_image).images[0]
make_image_grid([init_image, depth_image, image_control_net], rows=1, cols=3)
```

<div class="flex flex-row gap-4">
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأولية</figcaption>
</div>
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth/resolve/main/images/control.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">صورة العمق</figcaption>
</div>
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-controlnet.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">صورة ControlNet</figcaption>
</div>
</div>

دعونا نطبق [نمطًا](https://huggingface.co/nitrosocke/elden-ring-diffusion) جديدًا على الصورة التي تم إنشاؤها من ControlNet عن طريق ربطها بخط أنابيب من الصورة إلى الصورة:

```py
pipeline = AutoPipelineForImage2Image.from_pretrained(
"nitrosocke/elden-ring-diffusion", torch_dtype=torch.float16,
)
pipeline.enable_model_cpu_offload()
# احذف السطر التالي إذا لم يكن xFormers مثبتًا أو إذا كان لديك PyTorch 2.0 أو أعلى مثبتًا
pipeline.enable_xformers_memory_efficient_attention()

prompt = "elden ring style astronaut in a jungle" # تضمين الرمز المميز "elden ring style" في النص الفعلي
negative_prompt = "ugly, deformed, disfigured, poor details, bad anatomy"

image_elden_ring = pipeline(prompt, negative_prompt=negative_prompt, image=image_control_net, strength=0.45, guidance_scale=10.5).images[0]
make_image_grid([init_image, depth_image, image_control_net, image_elden_ring], rows=2, cols=2)
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-elden-ring.png">
</div>

## تحسين

إن تشغيل نماذج الانتشار باهظ التكلفة وكثيف الاستخدام للحوسبة، ولكن باستخدام بعض الحيل التحسينية، من الممكن تمامًا تشغيلها على معالجات الرسومات GPU للمستهلكين والطبقة المجانية. على سبيل المثال، يمكنك استخدام شكل أكثر كفاءة من حيث الذاكرة للاهتمام مثل [اهتمام المنتج النقطي المُمَيز](../optimization/torch2.0#scaled-dot-product-attention) من PyTorch 2.0 أو [xFormers](../optimization/xformers) (يمكنك استخدام أحدهما، ولكن لا توجد حاجة لاستخدام كليهما). يمكنك أيضًا نقل النموذج إلى وحدة معالجة الرسومات GPU أثناء انتظار مكونات خط الأنابيب الأخرى على وحدة المعالجة المركزية CPU.

```diff
+ pipeline.enable_model_cpu_offload()
+ pipeline.enable_xformers_memory_efficient_attention()
```

مع [`torch.compile`](../optimization/torch2.0#torchcompile)، يمكنك زيادة سرعة الاستدلال لديك عن طريق لف UNet الخاص بك به:

```py
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
```

لمعرفة المزيد، اطلع على أدلة [تقليل استخدام الذاكرة](../optimization/memory) و [Torch 2.0](../optimization/torch2.0).