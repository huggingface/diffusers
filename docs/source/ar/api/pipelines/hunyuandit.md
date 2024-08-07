# Hunyuan-DiT

![Chinese Elements Understanding](https://github.com/gnobitab/diffusers-hunyuan/assets/1157982/39b99036-c3cb-4f16-bb1a-40ec25eda573)

[Hunyuan-DiT: محول انتشار متعدد الأنماط وقوي مع فهم دقيق للغة الصينية](https://arxiv.org/abs/2405.08748) من Tencent Hunyuan.

الملخص من الورقة هو:

*نحن نقدم Hunyuan-DiT، وهو محول انتشار نصي إلى صورة مع فهم دقيق لكل من اللغة الإنجليزية والصينية. ولبناء Hunyuan-DiT، قمنا بتصميم هيكل المحول اللغوي وترميز النص والترميز الموضعي بعناية. كما أننا قمنا ببناء خط أنابيب البيانات بالكامل من الصفر لتحديث وتقييم البيانات لتحسين النموذج التكراري. وللفهم الدقيق للغة، قمنا بتدريب نموذج اللغة متعددة الوسائط لتنقيح تعليقات الصور. وأخيراً، يمكن لـ Hunyuan-DiT إجراء حوار متعدد الوسائط متعدد الأدوار مع المستخدمين، وتوليد الصور وتنقيحها وفقًا للسياق. ومن خلال بروتوكول التقييم البشري الشامل لدينا الذي يضم أكثر من 50 مقيمًا بشريًا محترفًا، يحدد Hunyuan-DiT معيارًا جديدًا في مجال توليد الصور من اللغة الصينية مقارنة بالنماذج مفتوحة المصدر الأخرى.*

يمكنك العثور على كود المصدر الأصلي في [Tencent/HunyuanDiT](https://github.com/Tencent/HunyuanDiT) وجميع نقاط التفتيش المتاحة في [Tencent-Hunyuan](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT).

**المزايا**: يدعم HunyuanDiT الصينية/الإنجليزية إلى الصورة، وتوليد الصور متعددة الدقة.

يتكون HunyuanDiT من المكونات التالية:

* يستخدم محول انتشار كهيكل أساسي
* يجمع بين ترميزي نص، CLIP ثنائي اللغة وترميز T5 متعدد اللغات

<Tip>

تأكد من مراجعة دليل الجداول الزمنية [guide](../../using-diffusers/schedulers.md) لمعرفة كيفية استكشاف المقايضة بين سرعة الجدول الزمني والجودة، وقسم [إعادة استخدام المكونات عبر خطوط الأنابيب](../../using-diffusers/loading.md#reuse-a-pipeline) لمعرفة كيفية تحميل المكونات نفسها بكفاءة في خطوط أنابيب متعددة.

</Tip>

## التحسين

يمكنك تحسين وقت تشغيل خط الأنابيب واستهلاك الذاكرة باستخدام torch.compile و feed-forward chunking. لمعرفة المزيد عن طرق التحسين، راجع أدلة [تسريع الاستدلال](../../optimization/fp16) و [تقليل استخدام الذاكرة](../../optimization/memory).

### الاستدلال

استخدم [`torch.compile`](https://huggingface.co/docs/diffusers/main/en/tutorials/fast_diffusion#torchcompile) لتقليل زمن الاستدلال.

أولاً، قم بتحميل خط الأنابيب:

```python
from diffusers import HunyuanDiTPipeline
import torch

pipeline = HunyuanDiTPipeline.from_pretrained(
    "Tencent-Hunyuan/HunyuanDiT-Diffusers", torch_dtype=torch.float16
).to("cuda")
```

ثم قم بتغيير تخطيط الذاكرة لمكونات خط الأنابيب `transformer` و `vae` إلى `torch.channels-last`:

```python
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.vae.to(memory_format=torch.channels_last)
```

أخيرًا، قم بتجميع المكونات وتشغيل الاستدلال:

```python
pipeline.transformer = torch.compile(pipeline.transformer, mode="max-autotune", fullgraph=True)
pipeline.vae.decode = torch.compile(pipeline.vae.decode, mode="max-autotune", fullgraph=True)

image = pipeline(prompt="رائد فضاء يركب حصانًا").images[0]
```

فيما يلي نتائج الاختبار على جهاز 80GB A100:

```bash
With torch.compile(): Average inference time: 12.470 seconds.
Without torch.compile(): Average inference time: 20.570 seconds.
```

### تحسين الذاكرة

من خلال تحميل ترميز النص T5 في 8 بتات، يمكنك تشغيل خط الأنابيب في أقل من 6 جيجابايت من ذاكرة GPU VRAM. راجع [هذا السكريبت](https://gist.github.com/sayakpaul/3154605f6af05b98a41081aaba5ca43e) للحصول على التفاصيل.

علاوة على ذلك، يمكنك استخدام طريقة [`~HunyuanDiT2DModel.enable_forward_chunking`] لتقليل استخدام الذاكرة. يقوم التقطيع التلقائي بتشغيل طبقات التغذية الأمامية في كتلة محول في حلقة بدلاً من تشغيلها جميعًا مرة واحدة. يمنحك ذلك مقايضة بين استهلاك الذاكرة ووقت تشغيل الاستدلال.

```diff
+ pipeline.transformer.enable_forward_chunking(chunk_size=1, dim=1)
```

## HunyuanDiTPipeline

[[autodoc]] HunyuanDiTPipeline

- all
- __call__