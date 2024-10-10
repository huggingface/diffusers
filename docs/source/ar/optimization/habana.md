# Habana Gaudi

يتوافق 🤗 Diffusers مع Habana Gaudi من خلال 🤗 [Optimum]. اتبع [دليل التثبيت] لتثبيت برامج تشغيل SynapseAI وGaudi، ثم قم بتثبيت Optimum Habana:

```bash
python -m pip install --upgrade-strategy eager optimum[habana]
```

لإنشاء الصور باستخدام Stable Diffusion 1 و2 على Gaudi، يلزمك إنشاء مثيلين:
- [`~optimum.habana.diffusers.GaudiStableDiffusionPipeline`]، خط أنابيب لإنشاء الصور بناءً على النص.
- [`~optimum.habana.diffusers.GaudiDDIMScheduler`]، مخطط Gaudi الأمثل.

عند تهيئة خط الأنابيب، يجب عليك تحديد `use_habana=True` لنشره على HPUs وللحصول على أسرع عملية إنشاء، يجب عليك تمكين **رسوم HPU** مع `use_hpu_graphs=True`.

أخيرًا، حدد [`~optimum.habana.GaudiConfig`] الذي يمكن تنزيله من منظمة [Habana] على Hub.

```python
from optimum.habana import GaudiConfig
from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiStableDiffusionPipeline

model_name = "stabilityai/stable-diffusion-2-base"
scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
pipeline = GaudiStableDiffusionPipeline.from_pretrained(
    model_name,
    scheduler=scheduler,
    use_habana=True,
    use_hpu_graphs=True,
    gaudi_config="Habana/stable-diffusion-2",
)
```

الآن يمكنك استدعاء خط الأنابيب لإنشاء الصور على شكل دفعات بناءً على موجه واحد أو أكثر:

```python
outputs = pipeline(
    prompt=[
        "High quality photo of an astronaut riding a horse in space",
        "Face of a yellow cat, high resolution, sitting on a park bench",
    ],
    num_images_per_prompt=10,
    batch_size=4,
)
```

## Benchmark

لقد قمنا باختبار أداء الجيل الأول من Gaudi وGaudi2 من Habana باستخدام تكوينات [Habana/stable-diffusion] و[Habana/stable-diffusion-2] Gaudi (الدقة المختلطة bf16/fp32) لتوضيح أدائهما.

بالنسبة لـ [Stable Diffusion v1.5] على الصور بحجم 512x512:

|                        | الكمون (حجم الدفعة = 1) | الإنتاجية  |
| ---------------------- |:------------------------:|:---------------------------:|
| الجيل الأول من Gaudi  | 3.80 ثانية               | 0.308 صورة/ثانية (حجم الدفعة = 8)             |
| Gaudi2                 | 1.33 ثانية               | 1.081 صورة/ثانية (حجم الدفعة = 8)             |

بالنسبة لـ [Stable Diffusion v2.1] على الصور بحجم 768x768:

|                        | الكمون (حجم الدفعة = 1) | الإنتاجية                      |
| ---------------------- |:------------------------:|:-------------------------------:|
| الجيل الأول من Gaudi  | 10.2 ثانية              | 0.108 صورة/ثانية (حجم الدفعة = 4) |
| Gaudi2                 | 3.17 ثانية              | 0.379 صورة/ثانية (حجم الدفعة = 8) |