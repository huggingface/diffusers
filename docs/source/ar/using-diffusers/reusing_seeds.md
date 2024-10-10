# خطوط أنابيب قابلة للاستنساخ

تعد نماذج الانتشار عشوائية بطبيعتها، مما يسمح لها بتوليد مخرجات مختلفة في كل مرة يتم تشغيلها. ولكن هناك أوقات معينة تريد فيها توليد نفس الإخراج في كل مرة، مثل عند الاختبار وتكرار النتائج، وحتى عند [تحسين جودة الصورة](#deterministic-batch-generation). في حين أنه لا يمكنك توقع الحصول على نتائج متطابقة عبر المنصات، يمكنك توقع نتائج قابلة للاستنساخ عبر الإصدارات والمنصات ضمن نطاق تسامح معين (على الرغم من أن حتى هذا قد يختلف).

سيوضح هذا الدليل كيفية التحكم في العشوائية للجيل الحتمي على وحدة المعالجة المركزية ووحدات معالجة الرسوميات.

> [!TIP]
> نوصي بشدة بقراءة بيان PyTorch [عن قابلية الاستنساخ](https://pytorch.org/docs/stable/notes/randomness.html):
>
> "النتائج القابلة للاستنساخ تمامًا غير مضمونة عبر إصدارات PyTorch أو الالتزامات الفردية أو المنصات المختلفة. علاوة على ذلك، قد لا تكون النتائج قابلة للاستنساخ بين عمليات التنفيذ على وحدة المعالجة المركزية ووحدات معالجة الرسوميات، حتى عند استخدام البذور المتطابقة."

## التحكم في العشوائية

خلال الاستدلال، تعتمد خطوط الأنابيب اعتمادًا كبيرًا على عمليات أخذ العينات العشوائية التي تشمل إنشاء
تنسورات ضوضاء غاوسيان لإزالة التشويش وإضافة ضوضاء إلى خطوة الجدولة.

الق نظرة على قيم التنسور في [`DDIMPipeline`] بعد خطوتين من الاستدلال.

```python
from diffusers import DDIMPipeline
import numpy as np

ddim = DDIMPipeline.from_pretrained("google/ddpm-cifar10-32", use_safetensors=True)
image = ddim(num_inference_steps=2, output_type="np").images
print(np.abs(image).sum())
```

يعرض تشغيل التعليمات البرمجية أعلاه قيمة واحدة، ولكن إذا قمت بتشغيله مرة أخرى، فستحصل على قيمة مختلفة.

في كل مرة يتم فيها تشغيل خط الأنابيب، يستخدم [torch.randn](https://pytorch.org/docs/stable/generated/torch.randn.html) بذرة عشوائية مختلفة لإنشاء تنسورات ضوضاء غاوس. يؤدي هذا إلى نتيجة مختلفة في كل مرة يتم تشغيله فيها ويمكّن خط أنابيب الانتشار من إنشاء صورة عشوائية مختلفة في كل مرة.

ولكن إذا كنت بحاجة إلى توليد نفس الصورة بشكل موثوق، فإن ذلك يعتمد على ما إذا كنت تشغل خط الأنابيب على وحدة المعالجة المركزية أو وحدة معالجة الرسوميات.

> [!TIP]
> قد يبدو من غير البديهي تمرير كائنات `Generator` إلى خط الأنابيب بدلاً من قيمة العدد الصحيح التي تمثل البذرة. ومع ذلك، هذا هو التصميم الموصى به عند العمل مع النماذج الاحتمالية في PyTorch لأن `Generator` هو *حالة عشوائية* يمكن تمريرها إلى خطوط أنابيب متعددة في تسلسل. بمجرد استهلاك `Generator`، يتم تغيير *الحالة* في المكان، مما يعني أنه حتى إذا قمت بتمرير نفس `Generator` إلى خط أنابيب مختلف، فلن ينتج نفس النتيجة لأن الحالة قد تغيرت بالفعل.

<hfoptions id="hardware">
<hfoption id="CPU">

لإنشاء نتائج قابلة للاستنساخ على وحدة المعالجة المركزية، ستحتاج إلى استخدام [Generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) PyTorch وتعيين بذرة. الآن عند تشغيل التعليمات البرمجية، فهو يطبع دائمًا قيمة `1491.1711` لأن كائن `Generator` بالبذرة يتم تمريره إلى جميع الوظائف العشوائية في خط الأنابيب. يجب أن تحصل على نتيجة مماثلة، إن لم تكن متطابقة، على أي أجهزة وأي إصدار PyTorch الذي تستخدمه.

```python
import torch
import numpy as np
from diffusers import DDIMPipeline

ddim = DDIMPipeline.from_pretrained("google/ddpm-cifar10-32", use_safetensors=True)
generator = torch.Generator(device="cpu").manual_seed(0)
image = ddim(num_inference_steps=2, output_type="np", generator=generator).images
print(np.abs(image).sum())
```

</hfoption>
<hfoption id="GPU">

إن كتابة خط أنابيب قابل للاستنساخ على وحدة معالجة الرسوميات أكثر تعقيدًا بعض الشيء، ولا تُضمن قابلية الاستنساخ الكاملة عبر الأجهزة المختلفة لأن الضرب المصفوفي - الذي تتطلبه خطوط أنابيب الانتشار الكثير منه - أقل حتمية على وحدة معالجة الرسوميات منه على وحدة المعالجة المركزية. على سبيل المثال، إذا قمت بتشغيل نفس مثال التعليمات البرمجية من مثال وحدة المعالجة المركزية، فستحصل على نتيجة مختلفة على الرغم من تطابق البذور. ويرجع ذلك إلى أن وحدة معالجة الرسوميات تستخدم مولد أرقام عشوائية مختلف عن وحدة المعالجة المركزية.

```python
import torch
import numpy as np
from diffusers import DDIMPipeline

ddim = DDIMPipeline.from_pretrained("google/ddpm-cifar10-32", use_safetensors=True)
ddim.to("cuda")
generator = torch.Generator(device="cuda").manual_seed(0)
image = ddim(num_inference_steps=2, output_type="np", generator=generator).images
print(np.abs(image).sum())
```

للتغلب على هذه المشكلة، تحتوي Diffusers على وظيفة [`~utils.torch_utils.randn_tensor`] لإنشاء ضوضاء عشوائية على وحدة المعالجة المركزية، ثم نقل التنسور إلى وحدة معالجة الرسوميات إذا لزم الأمر. يتم استخدام وظيفة [`~utils.torch_utils.randn_tensor`] في كل مكان داخل خط الأنابيب. الآن يمكنك استدعاء [torch.manual_seed](https://pytorch.org/docs/stable/generated/torch.manual_seed.html) الذي يقوم تلقائيًا بإنشاء `Generator` وحدة المعالجة المركزية التي يمكن تمريرها إلى خط الأنابيب حتى إذا كان قيد التشغيل على وحدة معالجة الرسوميات.

```python
import torch
import numpy as np
from diffusers import DDIMPipeline

ddim = DDIMPipeline.from_pretrained("google/ddpm-cifar10-32", use_safetensors=True)
ddim.to("cuda")
generator = torch.manual_seed(0)
image = ddim(num_inference_steps=2, output_type="np", generator=generator).images
print(np.abs(image).sum())
```

> [!TIP]
> إذا كانت قابلية الاستنساخ مهمة لحالتك الاستخدام، فنحن نوصي دائمًا بتمرير `Generator` وحدة المعالجة المركزية. غالبًا ما يكون فقدان الأداء ضئيلًا، وستولد قيمًا أكثر تشابهاً مما لو تم تشغيل خط الأنابيب على وحدة معالجة الرسوميات.

أخيرًا، غالبًا ما تكون خطوط الأنابيب الأكثر تعقيدًا مثل [`UnCLIPPipeline`] عرضة للغاية
خطأ انتشار الدقة. ستحتاج إلى استخدام
بالضبط نفس الأجهزة وإصدار PyTorch لقابلية الاستنساخ الكاملة.

</hfoption>
</hfoptions>

## الخوارزميات الحتمية

يمكنك أيضًا تكوين PyTorch لاستخدام خوارزميات حتمية لإنشاء خط أنابيب قابل للاستنساخ. الجانب السلبي هو أن الخوارزميات الحتمية قد تكون أبطأ من الخوارزميات غير الحتمية، وقد تلاحظ انخفاضًا في الأداء.

يحدث السلوك غير الحتمي عندما يتم إطلاق العمليات في أكثر من تدفق CUDA واحد. لتجنب ذلك، قم بتعيين متغير البيئة [CUBLAS_WORKSPACE_CONFIG](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility) إلى `:16:8` لاستخدام حجم مخزن مؤقت واحد فقط أثناء وقت التشغيل.

عادةً ما يقوم PyTorch باختبار خوارزميات متعددة لاختيار أسرعها، ولكن إذا كنت تريد قابلية الاستنساخ، فيجب عليك تعطيل هذه الميزة لأن الاختبار قد يختار خوارزميات مختلفة في كل مرة. قم بتعيين [enable_full_determinism](https://github.com/huggingface/diffusers/blob/142f353e1c638ff1d20bd798402b68f72c1ebbdd/src/diffusers/utils/testing_utils.py#L861) Diffusers لتمكين الخوارزميات الحتمية.

```py
enable_full_determinism()
```

الآن عند تشغيل نفس خط الأنابيب مرتين، ستحصل على نتائج متطابقة.

```py
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
g = torch.Generator(device="cuda")

prompt = "A bear is playing a guitar on Times Square"

g.manual_seed(0)
result1 = pipe(prompt=prompt, num_inference_steps=50, generator=g, output_type="latent").images

g.manual_seed(0)
result2 = pipe(prompt=prompt, num_inference_steps=50, generator=g, output_type="latent").images

print("L_inf dist =", abs(result1 - result2).max())
"L_inf dist = tensor(0., device='cuda:0')"
```

## توليد دفعات حتمية

تتمثل إحدى التطبيقات العملية لإنشاء خطوط أنابيب قابلة للاستنساخ في *توليد دفعات حتمية*. تقوم بتوليد دفعة من الصور واختيار صورة واحدة لتحسينها باستخدام موجه أكثر تفصيلاً. الفكرة الرئيسية هي تمرير قائمة من [Generator's](https://pytorch.org/docs/stable/generated/torch.Generator.html) إلى خط الأنابيب وربط كل `Generator` ببذرة حتى تتمكن من إعادة استخدامها.

دعنا نستخدم نقطة تفتيش [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) وإنشاء دفعة من الصور.

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import make_image_grid

pipeline = DiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
pipeline = pipeline.to("cuda")
```

قم بتعريف أربعة `Generator` مختلفة وقم بتعيين بذرة لكل `Generator` (`0` إلى `3`). ثم قم بتوليد دفعة من الصور واختر واحدة للعمل عليها.

> [!WARNING]
> استخدم تعبير قائمة يفحص عبر حجم الدفعة المحدد في `range()` لإنشاء كائن `Generator` فريد لكل صورة في الدفعة. إذا قمت بضرب `Generator` بعدد صحيح لحجم الدفعة، فلن يتم إنشاء سوى *كائن Generator* واحد يتم استخدامه بشكل تسلسلي لكل صورة في الدفعة.
>
> ```py
> [torch.Generator().manual_seed(seed)] * 4
> ```

```python
generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(4)]
prompt = "Labrador in the style of Vermeer"
images = pipeline(prompt, generator=generator, num_images_per_prompt=4).images[0]
make_image_grid(images, rows=2, cols=2)
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/reusabe_seeds.jpg"/>
</div>

دعنا نحسن الصورة الأولى (يمكنك اختيار أي صورة تريدها) والتي تتوافق مع `Generator` بالبذرة `0`. أضف بعض النص الإضافي إلى موجهك، ثم تأكد من إعادة استخدام نفس `Generator` بالبذرة `0`. يجب أن تشبه جميع الصور المولدة الصورة الأولى.

```python
prompt = [prompt + t for t in [", highly realistic", ", artsy", ", trending", ", colorful"]]
generator = [torch.Generator(device="cuda").manual_seed(0) for i in range(4)]
images = pipeline(prompt, generator=generator).images
make_image_grid(images, rows=2, cols=2)
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/reusabe_seeds_2.jpg"/>
</div>