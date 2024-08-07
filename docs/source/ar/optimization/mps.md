# Metal Performance Shaders (MPS)
🤗 Diffusers متوافق مع Apple silicon (M1/M2 chips) باستخدام PyTorch [`mps`](https://pytorch.org/docs/stable/notes/mps.html) device، والذي يستخدم Metal framework للاستفادة من GPU على أجهزة MacOS. ستحتاج إلى ما يلي:

- جهاز كمبيوتر macOS بمعمارية Apple silicon (M1/M2)
- نظام macOS 12.6 أو أحدث (يوصى بـ 13.0 أو أحدث)
- إصدار arm64 من Python
- [PyTorch 2.0](https://pytorch.org/get-started/locally/) (موصى به) أو 1.13 (الإصدار الأدنى المدعوم لـ `mps`)

يستخدم backend `mps` واجهة PyTorch `.to()` لنقل خط أنابيب Stable Diffusion إلى جهاز M1 أو M2 الخاص بك:

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")

# يوصى به إذا كان لدى جهاز الكمبيوتر الخاص بك <64 جيجابايت من ذاكرة الوصول العشوائي
pipe.enable_attention_slicing()

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
image
```

<Tip warning={true}>

قد يؤدي إنشاء عدة موجهات في دفعة واحدة إلى [التوقف](https://github.com/huggingface/diffusers/issues/363) أو الفشل في العمل بشكل موثوق. نعتقد أن هذا يرتبط بـ [`mps`](https://github.com/pytorch/pytorch/issues/84039) backend في PyTorch. بينما يتم التحقيق في هذا الأمر، يجب عليك التكرار بدلاً من الدفعات.

</Tip>

إذا كنت تستخدم **PyTorch 1.13**، فيجب عليك "تهيئة" خط الأنابيب بمرور إضافي لمرة واحدة من خلاله. هذا حل مؤقت لمشكلة حيث تنتج أول عملية استدلال نتائج مختلفة قليلاً عن العمليات اللاحقة. تحتاج فقط إلى إجراء هذه الخطوة مرة واحدة، وبعد خطوة الاستدلال الواحدة فقط، يمكنك تجاهل النتيجة.

```diff
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("mps")
pipe.enable_attention_slicing()

prompt = "a photo of an astronaut riding a horse on mars"
# First-time "warmup" pass if PyTorch version is 1.13
+ _ = pipe(prompt, num_inference_steps=1)

# The results match those from the CPU device after the warmup pass.
image = pipe(prompt).images[0]
```

## استكشاف الأخطاء وإصلاحها

تتأثر أداء M1/M2 بشكل كبير بضغط الذاكرة. عندما يحدث ذلك، يقوم النظام تلقائيًا بالتبديل إذا لزم الأمر، مما يتسبب في تدهور الأداء بشكل كبير.

لمنع حدوث ذلك، نوصي بـ *attention slicing* للحد من ضغط الذاكرة أثناء الاستدلال ومنع التبديل. هذا أمر مهم بشكل خاص إذا كان لدى جهاز الكمبيوتر الخاص بك أقل من 64 جيجابايت من ذاكرة الوصول العشوائي للنظام، أو إذا كنت تقوم بتوليد صور بدقة أكبر من 512×512 بكسل. قم بالاتصال بـ [`~DiffusionPipeline.enable_attention_slicing`] function على خط الأنابيب الخاص بك:

```py
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("mps")
pipeline.enable_attention_slicing()
```

تقوم Attention slicing بعملية الانتباه المكلفة في عدة خطوات بدلاً من القيام بها جميعًا مرة واحدة. وعادة ما يحسن الأداء بنسبة ~20% في أجهزة الكمبيوتر بدون ذاكرة عالمية، ولكننا لاحظنا *أداء أفضل* في معظم أجهزة Apple silicon ما لم يكن لديك 64 جيجابايت من ذاكرة الوصول العشوائي أو أكثر.