# Safe Stable Diffusion

تم اقتراح Safe Stable Diffusion في [Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models](https://huggingface.co/papers/2211.05105) ويقلل من التدهور غير المناسب من نماذج Stable Diffusion لأنها مدربة على مجموعات بيانات غير مصفاة تم جمعها من الويب. على سبيل المثال، قد يقوم Stable Diffusion بتوليد صور عارية أو عنيفة أو صور تصور إيذاء النفس أو محتوى مسيء بشكل غير متوقع. Safe Stable Diffusion هو امتداد لـ Stable Diffusion يقلل بشكل كبير من هذا النوع من المحتوى.

الملخص من الورقة هو:

*حققت نماذج توليد الصور المشروطة بالنص مؤخرًا نتائج مذهلة في جودة الصورة ومحاذاة النص، وبالتالي يتم استخدامها في عدد متزايد بسرعة من التطبيقات. ونظرًا لأنها تعتمد بشكل كبير على البيانات، وتعتمد على مجموعات بيانات بحجم مليار تم جمعها عشوائيًا من الإنترنت، فإنها تعاني أيضًا، كما نثبت، من سلوك بشري متدهور ومنحاز. بدورها، قد تعزز هذه النماذج حتى هذه التحيزات. للمساعدة في مكافحة هذه الآثار الجانبية غير المرغوب فيها، نقدم الانتشار الآمن للانتشار (SLD). على وجه التحديد، لقياس التدهور غير المناسب بسبب مجموعات التدريب غير المفلترة وغير المتوازنة، أنشأنا بيئة اختبار لتوليد الصور - موجهات الصور غير اللائقة (I2P) - تحتوي على موجهات صورة واقعية مخصصة لتغطية مفاهيم مثل العري والعنف. وكما يظهر تقييمنا التجريبي الشامل، فإن SLD الذي تم تقديمه يزيل الأجزاء غير المناسبة من الصورة أثناء عملية الانتشار، دون الحاجة إلى تدريب إضافي ودون أي تأثير سلبي على جودة الصورة أو محاذاة النص بشكل عام.*

## نصائح

استخدم خاصية `safety_concept` من [`StableDiffusionPipelineSafe`] للتحقق من مفهوم الأمان الحالي وتحريره:

```python
>>> from diffusers import StableDiffusionPipelineSafe

>>> pipeline = StableDiffusionPipelineSafe.from_pretrained("AIML-TUDA/stable-diffusion-safe")
>>> pipeline.safety_concept
'an image showing hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, drug use, theft, vandalism, weapons, child abuse, brutality, cruelty'
```

بالنسبة لكل توليد صورة، يتم أيضًا تضمين المفهوم النشط في [`StableDiffusionSafePipelineOutput`].

هناك 4 تكوينات (`SafetyConfig.WEAK`، `SafetyConfig.MEDIUM`، `SafetyConfig.STRONG`، و`SafetyConfig.MAX`) التي يمكن تطبيقها:

```python
>>> from diffusers import StableDiffusionPipelineSafe
>>> from diffusers.pipelines.stable_diffusion_safe import SafetyConfig

>>> pipeline = StableDiffusionPipelineSafe.from_pretrained("AIML-TUDA/stable-diffusion-safe")
>>> prompt = "the four horsewomen of the apocalypse, painting by the artists Tom of Finland, Gaston Bussiere, Craig Mullins, and J. C. Leyendecker"
>>> out = pipeline(prompt=prompt, **SafetyConfig.MAX)
```

<Tip>

تأكد من الاطلاع على قسم نصائح Stable Diffusion [Tips](overview#tips) لمعرفة كيفية استكشاف المقايضة بين سرعة الجدولة والجودة، وكيفية إعادة استخدام مكونات الأنابيب بكفاءة!

</Tip>

## StableDiffusionPipelineSafe

[[autodoc]] StableDiffusionPipelineSafe
- all
- __call__

## StableDiffusionSafePipelineOutput

[[autodoc]] pipelines.stable_diffusion_safe.StableDiffusionSafePipelineOutput
- all
- __call__