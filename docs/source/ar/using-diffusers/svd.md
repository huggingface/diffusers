# Stable Video Diffusion

[[open-in-colab]]

[Stable Video Diffusion (SVD)](https://huggingface.co/papers/2311.15127) هو نموذج قوي لتوليد الفيديو بناءً على صورة، ويمكنه توليد فيديو عالي الدقة (576x1024) لمدة 2-4 ثوانٍ بناءً على صورة مدخلة.

سيوضح هذا الدليل كيفية استخدام SVD لتوليد مقاطع فيديو قصيرة من الصور.

قبل البدء، تأكد من تثبيت المكتبات التالية:

```py
!pip install -q -U diffusers transformers accelerate
```

هناك متغيران من هذا النموذج، [SVD](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) و [SVD-XT](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt). تم تدريب نقطة تفتيش SVD لتوليد 14 إطارًا، وتم ضبط نقطة تفتيش SVD-XT بشكل إضافي لتوليد 25 إطارًا.

ستستخدم نقطة تفتيش SVD-XT لهذا الدليل.

```python
import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
"stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

# تحميل صورة التكييف
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

export_to_video(frames, "generated.mp4", fps=7)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">"الصورة المصدر لصاروخ"</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/output_rocket.gif"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">"الفيديو المولد من صورة المصدر"</figcaption>
</div>
</div>

## torch.compile

يمكنك تحقيق زيادة في السرعة بنسبة 20-25٪ على حساب زيادة طفيفة في الذاكرة عن طريق [تجميع](../optimization/torch2.0#torchcompile) شبكة UNet.

```diff
- pipe.enable_model_cpu_offload()
+ pipe.to("cuda")
+ pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
```

## تقليل استخدام الذاكرة

يتطلب إنشاء الفيديو قدرًا كبيرًا من الذاكرة لأنك تقوم بتوليد `num_frames` في وقت واحد، على غرار توليد الصور النصية بدفعة كبيرة. ولتقليل متطلبات الذاكرة، هناك خيارات متعددة تُوازن بين سرعة الاستدلال ومتطلبات الذاكرة المنخفضة:

- تمكين النقل إلى وحدة المعالجة المركزية: يتم نقل كل مكون من خط الأنابيب إلى وحدة المعالجة المركزية بمجرد عدم الحاجة إليه بعد الآن.
- تمكين التقطيع للأمام: تعمل طبقة التغذية الأمامية في حلقة بدلاً من تشغيل تغذية أمامية واحدة بحجم دفعة كبير.
- تقليل `decode_chunk_size`: يقوم VAE بفك تشفير الإطارات على شكل قطع بدلاً من فك تشفيرها جميعًا معًا. يؤدي تعيين `decode_chunk_size=1` إلى فك تشفير إطار واحد في كل مرة ويستخدم أقل قدر من الذاكرة (نوصي بتعديل هذه القيمة بناءً على ذاكرة GPU المتوفرة)، ولكن قد يكون للفيديو بعض الوميض.

```diff
- pipe.enable_model_cpu_offload()
- frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]
+ pipe.enable_model_cpu_offload()
+ pipe.unet.enable_forward_chunking()
+ frames = pipe(image, decode_chunk_size=2, generator=generator, num_frames=25).frames[0]
```

من المفترض أن تقلل هذه الحيل مجتمعة متطلبات الذاكرة إلى أقل من 8 جيجابايت من ذاكرة الوصول العشوائي الديناميكي.

## التكييف الدقيق

يقبل Stable Diffusion Video أيضًا التكييف الدقيق، بالإضافة إلى صورة التكييف، مما يتيح مزيدًا من التحكم في الفيديو المولد:

- `fps`: عدد الإطارات في الثانية للفيديو المولد.
- `motion_bucket_id`: معرّف دلو الحركة الذي سيتم استخدامه للفيديو المولد. يمكن استخدام هذا للتحكم في حركة الفيديو المولد. يؤدي زيادة معرّف دلو الحركة إلى زيادة حركة الفيديو المولد.
- `noise_aug_strength`: مقدار الضوضاء المضافة إلى صورة التكييف. كلما زادت القيم، قل تشابه الفيديو مع صورة التكييف. كما أن زيادة هذه القيمة تزيد من حركة الفيديو المولد.

على سبيل المثال، لتوليد فيديو ذو حركة أكبر، استخدم معلمات التكييف الدقيق `motion_bucket_id` و`noise_aug_strength`:

```python
import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
"stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

# تحميل صورة التكييف
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator, motion_bucket_id=180, noise_aug_strength=0.1).frames[0]
export_to_video(frames, "generated.mp4", fps=7)
```

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/output_rocket_with_conditions.gif)