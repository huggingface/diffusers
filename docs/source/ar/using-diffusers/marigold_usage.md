# أنابيب Marigold لمهام رؤية الكمبيوتر

[Marigold](../api/pipelines/marigold) هي طريقة جديدة للتنبؤ الكثيف القائم على الانتشار، ومجموعة من الأنابيب لمهام رؤية الكمبيوتر المختلفة، مثل تقدير العمق الأحادي. سيُظهر لك هذا الدليل كيفية استخدام Marigold للحصول على تنبؤات سريعة وعالية الجودة للصور ومقاطع الفيديو.

تدعم كل قناة واحدة من مهام رؤية الكمبيوتر، والتي تأخذ كإدخال صورة RGB وتنتج *تنبؤًا* بالنمط الذي يهمك، مثل خريطة العمق لصورة الإدخال.

حاليًا، يتم تنفيذ المهام التالية:

| الأنبوب                                                                                                                                    | الأنماط المتوقعة                                                                                             |                                                                       العروض التوضيحية                                                                       |
|---------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------:|
| [MarigoldDepthPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/marigold/pipeline_marigold_depth.py)     | [العمق](https://en.wikipedia.org/wiki/Depth_map)، [التباين](https://en.wikipedia.org/wiki/Binocular_disparity) | [العرض التوضيحي السريع (LCM)](https://huggingface.co/spaces/prs-eth/marigold-lcm)، [العرض التوضيحي الأصلي البطيء (DDIM)](https://huggingface.co/spaces/prs-eth/marigold) |
| [MarigoldNormalsPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/marigold/pipeline_marigold_normals.py) | [المستعارات السطحية](https://en.wikipedia.org/wiki/Normal_mapping)                                                  |                                   [العرض التوضيحي السريع (LCM)](https://huggingface.co/spaces/prs-eth/marigold-normals-lcm)                                    |

يمكن العثور على نقاط التفتيش الأصلية في [PRS-ETH](https://huggingface.co/prs-eth/) منظمة Hugging Face.

يُقصد بنقاط التفتيش هذه العمل مع أنابيب الناشرات و [قاعدة التعليمات البرمجية الأصلية](https://github.com/prs-eth/marigold).

يمكن أيضًا استخدام التعليمات البرمجية الأصلية لتدريب نقاط تفتيش جديدة.

| نقطة التفتيش                                                                                    | النمط | التعليق                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|-----------------------------------------------------------------------------------------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [prs-eth/marigold-v1-0](https://huggingface.co/prs-eth/marigold-v1-0)                         | العمق    | أول نقطة تفتيش لعمق Marigold، والتي تتنبأ بخرائط العمق *الدالة على التكافؤ*. تمت دراسة أداء نقطة التفتيش هذه في المعايير في الورقة [الأصلية](https://huggingface.co/papers/2312.02145). تم تصميمه لاستخدامه مع `DDIMScheduler` أثناء الاستدلال، فهو يتطلب ما لا يقل عن 10 خطوات للحصول على تنبؤات موثوقة. يتراوح التنبؤ بالعمق الدالة على التكافؤ بين القيم في كل بكسل بين 0 (الطائرة القريبة) و1 (الطائرة البعيدة)؛ تختار كل من الطائرتين النموذج كجزء من عملية الاستدلال. راجع مرجع `MarigoldImageProcessor` للحصول على برامج مساعدة للتصور. |
| [prs-eth/marigold-depth-lcm-v1-0](https://huggingface.co/prs-eth/marigold-depth-lcm-v1-0)     | العمق    | نقطة تفتيش Marigold Depth السريعة، والتي تم ضبط دقتها من `prs-eth/marigold-v1-0`. تم تصميمه لاستخدامه مع `LCMScheduler` أثناء الاستدلال، فهو يتطلب خطوة واحدة فقط للحصول على تنبؤات موثوقة. تصل موثوقية التنبؤ إلى التشبع عند 4 خطوات وتنخفض بعد ذلك.                                                                                                                                                                                                                                                                                                                           |
| [prs-eth/marigold-normals-v0-1](https://huggingface.co/prs-eth/marigold-normals-v0-1)         | Normals  | نقطة تفتيش معاينة لأنبوب Marigold Normals. تم تصميمه لاستخدامه مع `DDIMScheduler` أثناء الاستدلال، فهو يتطلب ما لا يقل عن 10 خطوات للحصول على تنبؤات موثوقة. تنبؤات المستعارات السطحية هي متجهات ثلاثية الأبعاد ذات طول وحدة بقيم في النطاق من -1 إلى 1. *سيتم إيقاف هذه النقطة بعد إصدار إصدار `v1-0`.*                                                                                                                                                                                                                                              |
| [prs-eth/marigold-normals-lcm-v0-1](https://huggingface.co/prs-eth/marigold-normals-lcm-v0-1) | Normals  | نقطة تفتيش Marigold Normals السريعة، والتي تم ضبط دقتها من `prs-eth/marigold-normals-v0-1`. تم تصميمه لاستخدامه مع `LCMScheduler` أثناء الاستدلال، فهو يتطلب خطوة واحدة فقط للحصول على تنبؤات موثوقة. تصل موثوقية التنبؤ إلى التشبع عند 4 خطوات وتنخفض بعد ذلك. *سيتم إيقاف هذه النقطة بعد إصدار إصدار `v1-0`.*                                                                                                                                                                                                                                       |

تُعطى الأمثلة أدناه في الغالب للتنبؤ بالعمق، ولكن يمكن تطبيقها عالميًا مع الأنماط الأخرى المدعومة.

نحن نعرض التنبؤات باستخدام نفس صورة المدخلات لألبرت أينشتاين التي تم إنشاؤها بواسطة Midjourney.

يجعل هذا من السهل مقارنة تصورات التنبؤات عبر مختلف الأنماط ونقاط التفتيش.

<div class="flex gap-4" style="justify-content: center; width: 100%;">
<div style="flex: 1 1 50%; max-width: 50%;">
<img class="rounded-xl" src="https://marigoldmonodepth.github.io/images/einstein.jpg"/>
<figcaption class="mt-1 text-center text-sm text-gray-500">
صورة الإدخال المثال لجميع أنابيب Marigold
</figcaption>
</div>
</div>

### بدء سريع للتنبؤ بالعمق

للحصول على أول تنبؤ بالعمق، قم بتحميل نقطة تفتيش `prs-eth/marigold-depth-lcm-v1-0` في خط أنابيب `MarigoldDepthPipeline`، ومرر الصورة عبر الأنبوب، واحفظ التنبؤات:

```python
import diffusers
import torch

pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
"prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
).to("cuda")

image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
depth = pipe(image)

vis = pipe.image_processor.visualize_depth(depth.prediction)
vis[0].save("einstein_depth.png")

depth_16bit = pipe.image_processor.export_depth_to_16bit_png(depth.prediction)
depth_16bit[0].save("einstein_depth_16bit.png")
```

تقوم دالة التصور للعمق [`~pipelines.marigold.marigold_image_processing.MarigoldImageProcessor.visualize_depth`] بتطبيق أحد [مخططات الألوان في matplotlib](https://matplotlib.org/stable/users/explain/colors/colormaps.html) (`Spectral` بشكل افتراضي) لرسم خريطة للقيم البكسل المتوقعة من نطاق العمق أحادي القناة `[0، 1]` إلى صورة RGB.

مع مخطط الألوان "Spectral"، يتم طلاء البكسلات ذات العمق القريب باللون الأحمر، ويتم تعيين البكسلات البعيدة باللون الأزرق.

يتم تخزين ملف PNG ذو 16 بت القناة الفردية التي يتم رسمها خطيًا من النطاق `[0، 1]` إلى `[0، 65535]`.

فيما يلي التنبؤات الخام والمرئية؛ كما هو موضح، يسهل تمييز المناطق الداكنة (الشارب) في التصور:

<div class="flex gap-4">
<div style="flex: 1 1 50%; max-width: 50%;">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_einstein_lcm_depth_16bit.png"/>
<figcaption class="mt-1 text-center text-sm text-gray-500">
العمق المتوقع (PNG 16 بت)
</figcaption>
</div>
<div style="flex: 1 1 50%; max-width: 50%;">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_einstein_lcm_depth.png"/>
<figcaption class="mt-1 text-center text-sm text-gray-500">
تصور العمق المتوقع (طيفي)
</figcaption>
</div>
</div>

### بدء سريع للتنبؤ بالمستعارات السطحية

قم بتحميل نقطة تفتيش `prs-eth/marigold-normals-lcm-v0-1` في خط أنابيب `MarigoldNormalsPipeline`، ومرر الصورة عبر الأنبوب، واحفظ التنبؤات:

```python
import diffusers
import torch

pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(
"prs-eth/marigold-normals-lcm-v0-1", variant="fp16", torch_dtype=torch.float16
).to("cuda")

image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
normals = pipe(image)

vis = pipe.image_processor.visualize_normals(normals.prediction)
vis[0].save("einstein_normals.png")
```

تقوم دالة التصور للمستعارات [`~pipelines.marigold.marigold_image_processing.MarigoldImageProcessor.visualize_normals`] برسم خريطة للتنبؤ ثلاثي الأبعاد بقيم البكسل في النطاق `[-1، 1]` إلى صورة RGB.

تدعم دالة التصور عكس محاور المستعارات السطحية لجعل التصور متوافقًا مع خيارات أخرى لإطار المرجع.

مفهوميًا، يتم طلاء كل بكسل وفقًا لمتجه المستعار السطحي في إطار المرجع، حيث يشير محور `X` إلى اليمين، ويشير محور `Y` لأعلى، ويشير محور `Z` إلى المشاهد.

فيما يلي التنبؤ المرئي:

<div class="flex gap-4" style="justify-content: center; width: 100%;">
<div style="flex: 1 1 50%; max-width: 50%;">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_einstein_lcm_normals.png"/>
<figcaption class="mt-1 text-center text-sm text-gray-500">
تصور المستعارات السطحية المتوقعة
</figcaption>
</div>
</div>

في هذا المثال، من المؤكد أن طرف الأنف لديه نقطة على السطح، حيث يشير متجه المستعار السطحي مباشرة إلى المشاهد، مما يعني أن إحداثياته هي `[0، 0، 1]`.

يتم رسم هذا المتجه إلى RGB `[128، 128، 255]`، والذي يقابل اللون الأزرق البنفسجي.

وبالمثل، فإن المستعار السطحي على الخد في الجزء الأيمن من الصورة له مكون "X" كبير، مما يزيد من اللون الأحمر.

تعزز النقاط الموجودة على الكتفين والتي تشير إلى الأعلى بلون أخضر.
بالتأكيد، سأتبع تعليماتك وسأترجم فقط النص الموجود في الفقرات والعناوين، مع الحفاظ على الأكواد البرمجية والروابط والرموز الأخرى بلغتها الأصلية.

### تسريع الاستنتاج

تم بالفعل تحسين مقتطفات "البدء السريع" أعلاه للسرعة: فهي تحمل نقطة تفتيش LCM، وتستخدم متغير "fp16" للأوزان والحساب، وتؤدي خطوة واحدة فقط لإزالة التشويش بالانتشار.

يستغرق استدعاء "pipe(image)" 280 مللي ثانية على وحدة معالجة الرسوميات (GPU) من نوع RTX 3090.

داخلياً، يتم ترميز صورة الإدخال باستخدام مشفر VAE الخاص بـ Stable Diffusion، ثم تقوم شبكة U-Net بتنفيذ خطوة إزالة التشويش، وأخيراً، يتم فك تشفير التنبؤ الكامن باستخدام فك تشفير VAE إلى مساحة البكسل.

في هذه الحالة، يتم تخصيص اثنين من أصل ثلاثة استدعاءات للوحدات النمطية لتحويل بين مساحة البكسل والمساحة الكامنة لـ LDM.

نظرًا لأن المساحة الكامنة لـ Marigold متوافقة مع Stable Diffusion الأساسي، فمن الممكن تسريع استدعاء الأنابيب بأكثر من 3 مرات (85 مللي ثانية على RTX 3090) باستخدام [بديل خفيف لمشفر VAE الخاص بـ SD](../api/models/autoencoder_tiny):

كما هو مقترح في [التحسينات](../optimization/torch2.0#torch.compile)، قد يؤدي إضافة `torch.compile` إلى ضغط الأداء الإضافي اعتمادًا على الأجهزة المستهدفة:

## مقارنة نوعية مع Depth Anything

مع تحسينات السرعة أعلاه، يقدم Marigold تنبؤات أكثر تفصيلاً وأسرع من [Depth Anything](https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything) مع أكبر نقطة تفتيش [LiheYoung/depth-anything-large-hf](https://huggingface.co/LiheYoung/depth-anything-large-hf):

<div class="flex gap-4">
<div style="flex: 1 1 50%; max-width: 50%;">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_einstein_lcm_depth.png"/>
<figcaption class="mt-1 text-center text-sm text-gray-500">
Marigold LCM fp16 مع AutoEncoder صغير
</figcaption>
</div>
<div style="flex: 1 1 50%; max-width: 50%;">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/einstein_depthanything_large.png"/>
<figcaption class="mt-1 text-center text-sm text-gray-500">
Depth Anything كبير
</figcaption>
</div>
</div>

## زيادة الدقة وتجميع التنبؤات

تحتوي أنابيب Marigold على آلية تجميع مدمجة تجمع بين تنبؤات متعددة من أشباه عشوائية مختلفة.

هذه طريقة مباشرة لتحسين دقة التنبؤات، والاستفادة من الطبيعة التوليدية للانتشار.

يتم تنشيط مسار التجميع تلقائيًا عندما يتم تعيين وسيط "ensemble_size" إلى رقم أكبر من 1.

وعند السعي لتحقيق أقصى قدر من الدقة، من المنطقي ضبط "num_inference_steps" في نفس الوقت مع "ensemble_size".

تختلف القيم الموصى بها عبر نقاط التفتيش ولكنها تعتمد بشكل أساسي على نوع الجدولة.

يظهر تأثير التجميع بشكل جيد مع القواعد السطحية:

```python
import diffusers

model_path = "prs-eth/marigold-normals-v1-0"

model_paper_kwargs = {
diffusers.schedulers.DDIMScheduler: {
"num_inference_steps": 10,
"ensemble_size": 10,
},
diffusers.schedulers.LCMScheduler: {
"num_inference_steps": 4,
"ensemble_size": 5,
},
}

image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")

pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(model_path).to("cuda")
pipe_kwargs = model_paper_kwargs[type(pipe.scheduler)]

depth = pipe(image, **pipe_kwargs)

vis = pipe.image_processor.visualize_normals(depth.prediction)
vis[0].save("einstein_normals.png")
```

<div class="flex gap-4">
<div style="flex: 1 1 50%; max-width: 50%;">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_einstein_lcm_normals.png"/>
<figcaption class="mt-1 text-center text-sm text-gray-500">
القواعد السطحية، بدون تجميع
</figcaption>
</div>
<div style="flex: 1 1 50%; max-width: 50%;">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_einstein_normals.png"/>
<figcaption class="mt-1 text-center text-sm text-gray-500">
القواعد السطحية، مع التجميع
</figcaption>
</div>
</div>

كما هو موضح، حصلت جميع المناطق ذات البنى الدقيقة، مثل الشعر، على تنبؤات أكثر تحفظًا وأكثر صحة في المتوسط.

مثل هذه النتيجة أكثر ملاءمة للمهام الحساسة للدقة، مثل إعادة الإعمار ثلاثي الأبعاد.

## التقييم الكمي

لتقييم Marigold كميًا في لوحات القيادة القياسية والمعايير المرجعية (مثل NYU وKITTI، ومجموعات البيانات الأخرى)، اتبع بروتوكول التقييم الموصوف في الورقة: قم بتحميل نموذج الدقة الكاملة fp32 واستخدم القيم المناسبة لـ "num_inference_steps" و"ensemble_size".

قم بتعيين البذور العشوائية بشكل اختياري لضمان إمكانية إعادة الإنتاج. وستؤدي زيادة "batch_size" إلى زيادة استخدام الجهاز إلى الحد الأقصى.

```python
import diffusers
import torch

device = "cuda"
seed = 2024
model_path = "prs-eth/marigold-v1-0"

model_paper_kwargs = {
diffusers.schedulers.DDIMScheduler: {
"num_inference_steps": 50,
"ensemble_size": 10,
},
diffusers.schedulers.LCMScheduler: {
"num_inference_steps": 4,
"ensemble_size": 10,
},
}

image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")

generator = torch.Generator(device=device).manual_seed(seed)
pipe = diffusers.MarigoldDepthPipeline.from_pretrained(model_path).to(device)
pipe_kwargs = model_paper_kwargs[type(pipe.scheduler)]

depth = pipe(image, generator=generator, **pipe_kwargs)

# تقييم المقاييس
```

## استخدام عدم اليقين التنبئي

تستخدم آلية التجميع المدمجة في أنابيب Marigold عدة تنبؤات يتم الحصول عليها من أشباه عشوائية مختلفة.

كأثر جانبي، يمكن استخدامه لقياس عدم اليقين النظري (النموذجي)؛ ما عليك سوى تحديد "ensemble_size" أكبر من 1 وتعيين "output_uncertainty=True".

سيتم توفير عدم اليقين الناتج في حقل "uncertainty" للإخراج.

يمكن تصويره على النحو التالي:

```python
import diffusers
import torch

pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
"prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
).to("cuda")

image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
depth = pipe(
image,
ensemble_size=10,  # أي رقم أكبر من 1؛ تعطي القيم الأعلى دقة أعلى
output_uncertainty=True,
)

uncertainty = pipe.image_processor.visualize_uncertainty(depth.uncertainty)
uncertainty[0].save("einstein_depth_uncertainty.png")
```

<div class="flex gap-4">
<div style="flex: 1 1 50%; max-width: 50%;">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_einstein_depth_uncertainty.png"/>
<figcaption class="mt-1 text-center text-sm text-gray-500">
عدم اليقين في العمق
</figcaption>
</div>
<div style="flex: 1 1 50%; max-width: 50%;">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_einstein_normals_uncertainty.png"/>
<figcaption class="mt-1 text-center text-sm text-gray-500">
عدم اليقين في القواعد السطحية
</figcaption>
</div>
</div>

تفسير عدم اليقين سهل: تشير القيم الأعلى (البيضاء) إلى البكسلات التي يناضل فيها النموذج لتقديم تنبؤات متسقة.

من الواضح أن نموذج العمق أقل ثقة حول الحواف ذات الانقطاع، حيث يتغير عمق الكائن بشكل كبير.

ونموذج القواعد السطحية أقل ثقة في البنى الدقيقة، مثل الشعر، والمناطق الداكنة، مثل الياقة.

## معالجة الفيديو إطارًا تلو الآخر مع الاتساق الزمني

بسبب الطبيعة التوليدية لـ Marigold، فإن كل تنبؤ فريد ويتم تحديده بواسطة الضوضاء العشوائية التي تم أخذ عينات منها للتهيئة الكامنة.

يصبح هذا عيبًا واضحًا مقارنة بشبكات الانحدار الكثيفة من النهاية إلى النهاية، كما هو موضح في مقاطع الفيديو التالية:

<div class="flex gap-4">
<div style="flex: 1 1 50%; max-width: 50%;">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_obama.gif"/>
<figcaption class="mt-1 text-center text-sm text-gray-500">
فيديو الإدخال
</figcaption>
</div>
<div style="flex: 1 1 50%; max-width: 50%;">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_obama_depth_independent.gif"/>
<figcaption class="mt-1 text-center text-sm text-gray-500">
Marigold Depth المطبق على إطارات الفيديو بشكل مستقل
</figcaption>
</div>
</div>

لمعالجة هذه المشكلة، يمكن تمرير وسيط "latents" إلى الأنابيب، والذي يحدد نقطة البداية للانتشار.

وجدنا تجريبياً أن الجمع المحدب لنفس نقطة البداية الكامنة للضوضاء والكامنة التي تتوافق مع تنبؤ الإطار السابق يعطي نتائج سلسة بدرجة كافية، كما هو منفذ في المقتطف أدناه:

```python
import imageio
from PIL import Image
from tqdm import tqdm
import diffusers
import torch

device = "cuda"
path_in = "obama.mp4"
path_out = "obama_depth.gif"

pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
"prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
).to(device)
pipe.vae = diffusers.AutoencoderTiny.from_pretrained(
"madebyollin/taesd", torch_dtype=torch.float16
).to(device)
pipe.set_progress_bar_config(disable=True)

with imageio.get_reader(path_in) as reader:
size = reader.get_meta_data()['size']
last_frame_latent = None
latent_common = torch.randn(
(1, 4, 768 * size[1] // (8 * max(size)), 768 * size[0] // (8 * max(size)))
).to(device=device, dtype=torch.float16)

out = []
for frame_id, frame in tqdm(enumerate(reader), desc="Processing Video"):
frame = Image.fromarray(frame)
latents = latent_common
if last_frame_latent is not None:
latents = 0.9 * latents + 0.1 * last_frame_latent

depth = pipe(
frame, match_input_resolution=False, latents=latents, output_latent=True
)
last_frame_latent = depth.latent
out.append(pipe.image_processor.visualize_depth(depth.prediction)[0])

diffusers.utils.export_to_gif(out, path_out, fps=reader.get_meta_data()['fps'])
```

هنا، تبدأ عملية الانتشار من الكامنة المحسوبة المعطاة.

يحدد الأنبوب "output_latent=True" للوصول إلى "out.latent" ويحسب مساهمته في تهيئة الكامنة للإطار التالي.

النتيجة الآن أكثر استقرارًا:

<div class="flex gap-4">
<div style="flex: 1 1 50%; max-width: 50%;">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_obama_depth_independent.gif"/>
<figcaption class="mt-1 text-center text-sm text-gray-500">
Marigold Depth المطبق على إطارات الفيديو بشكل مستقل
</figcaption>
</div>
<div style="flex: 1 1 50%; max-width: 50%;">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_obama_depth_consistent.gif"/>
<figcaption class="mt-1 text-center text-sm text-gray-500">
Marigold Depth مع تهيئة الكامنة القسرية
</figcaption>
</div>
</div>

## Marigold for ControlNet

يعد التنبؤ بالعمق باستخدام نماذج الانتشار بالتزامن مع ControlNet تطبيقًا شائعًا جدًا.

وتلعب وضوح العمق دورًا حاسمًا في الحصول على نتائج عالية الجودة من ControlNet.

وكما هو موضح في المقارنات مع الطرق الأخرى أعلاه، يتفوق Marigold في تلك المهمة.

توضح القطعة أدناه كيفية تحميل صورة، وحساب العمق، وإدخالها في ControlNet بتنسيق متوافق:

```python
import torch
import diffusers

device = "cuda"
generator = torch.Generator(device=device).manual_seed(2024)
image = diffusers.utils.load_image(
"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_depth_source.png"
)

pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
"prs-eth/marigold-depth-lcm-v1-0", torch_dtype=torch.float16, variant="fp16"
).to(device)

depth_image = pipe(image, generator=generator).prediction
depth_image = pipe.image_processor.visualize_depth(depth_image, color_map="binary")
depth_image[0].save("motorcycle_controlnet_depth.png")

controlnet = diffusers.ControlNetModel.from_pretrained(
"diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16, variant="fp16"
).to(device)
pipe = diffusers.StableDiffusionXLControlNetPipeline.from_pretrained(
"SG161222/RealVisXL_V4.0", torch_dtype=torch.float16, variant="fp16", controlnet=controlnet
).to(device)
pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

controlnet_out = pipe(
prompt="high quality photo of a sports bike, city",
negative_prompt="",
guidance_scale=6.5,
num_inference_steps=25,
image=depth_image,
controlnet_conditioning_scale=0.7,
control_guidance_end=0.7,
generator=generator,
).images
controlnet_out[0].save("motorcycle_controlnet_out.png")
```

<div class="flex gap-4">
<div style="flex: 1 1 33%; max-width: 33%;">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_depth_source.png"/>
<figcaption class="mt-1 text-center text-sm text-gray-500">
صورة الإدخال
</figcaption>
</div>
<div style="flex: 1 1 33%; max-width: 33%;">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/motorcycle_controlnet_depth.png"/>
<figcaption class="mt-1 text-center text-sm text-gray-500">
العمق بتنسيق متوافق مع ControlNet
</figcaption>
</div>
<div style="flex: 1 1 33%; max-width: 33%;">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/motorcycle_controlnet_out.png"/>
<figcaption class="mt-1 text-center text-sm text-gray-500">
توليد ControlNet، المشروط بالعمق والملاحظة: "صورة عالية الجودة لدراجة رياضية، مدينة"
</figcaption>
</div>
</div>

نأمل أن تجد Marigold مفيدة لحل مهامك اللاحقة، سواء كانت جزءًا من سير عمل توليدي أوسع، أو مهمة إدراكية، مثل إعادة الإعمار ثلاثي الأبعاد.