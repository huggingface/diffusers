# DDIMScheduler

[Denoising Diffusion Implicit Models](https://huggingface.co/papers/2010.02502) (DDIM) بواسطة جيمينج سونج، وتشنلين مينغ وستيفانو إرمون.

ملخص الورقة البحثية هو:

*حققت النماذج الاحتمالية للتشتت المضاد للضوضاء (DDPMs) جودة عالية في توليد الصور دون تدريب الخصوم، إلا أنها تتطلب محاكاة سلسلة ماركوف لعدة خطوات لإنتاج عينة. ولتسريع عملية المعاينة، نقدم نماذج ضمنية للتشتت المضاد للضوضاء (DDIMs)، وهي فئة أكثر كفاءة من النماذج الاحتمالية التكرارية الضمنية التي لها نفس إجراء التدريب مثل DDPMs. في DDPMs، يتم تعريف عملية التوليد على أنها عكس عملية انتشار ماركوف. نقوم ببناء فئة من عمليات الانتشار غير الماركوفية التي تؤدي إلى نفس الهدف التدريبي، ولكن يمكن أن تكون عملية عكسها أسرع بكثير في المعاينة. نثبت تجريبياً أن نماذج DDIM يمكن أن تنتج عينات عالية الجودة أسرع 10x إلى 50x من حيث الوقت الفعلي مقارنة بـ DDPMs، مما يسمح لنا بالموازنة بين الحساب وجودة العينة، ويمكنها إجراء استيفاء صور ذي معنى دلالي مباشرة في الفراغ الدلالي.*

يمكن العثور على الشفرة الأساسية الأصلية لهذه الورقة البحثية على [ermongroup/ddim](https://github.com/ermongroup/ddim)، ويمكنك التواصل مع المؤلف على [tsong.me](https://tsong.me/).

## نصائح

تدعي الورقة البحثية [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) أن عدم التطابق بين إعدادات التدريب والاستدلال يؤدي إلى نتائج استدلال دون المستوى الأمثل لـ Stable Diffusion. ولحل هذه المشكلة، يقترح المؤلفون ما يلي:

<Tip warning={true}>
🧪 هذه ميزة تجريبية!
</Tip>

1. إعادة ضبط جدول الضوضاء لفرض نسبة إشارة إلى ضوضاء (SNR) نهائية تساوي صفرًا

```py
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, rescale_betas_zero_snr=True)
```

2. تدريب نموذج باستخدام `v_prediction` (أضف الحجة التالية إلى نصوص [train_text_to_image.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py) أو [train_text_to_image_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py))

```bash
--prediction_type="v_prediction"
```

3. تغيير المعاين لبدء العمل دائمًا من الخطوة الأخيرة

```py
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
```

4. إعادة ضبط التوجيه الخالي من التصنيف لمنع التعرض المفرط

```py
image = pipe(prompt, guidance_rescale=0.7).images[0]
```

على سبيل المثال:

```py
from diffusers import DiffusionPipeline, DDIMScheduler
import torch

pipe = DiffusionPipeline.from_pretrained("ptx0/pseudo-journey-v2", torch_dtype=torch.float16)
pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
)
pipe.to("cuda")

prompt = "A lion in galaxies, spirals, nebulae, stars, smoke, iridescent, intricate detail, octane render, 8k"
image = pipe(prompt, guidance_rescale=0.7).images[0]
image
```

## DDIMScheduler

[[autodoc]] DDIMScheduler

## DDIMSchedulerOutput

[[autodoc]] schedulers.scheduling_ddim.DDIMSchedulerOutput