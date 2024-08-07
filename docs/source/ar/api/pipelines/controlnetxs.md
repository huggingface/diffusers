لمزيد من المعلومات، راجع [دليل الجداول](../../using-diffusers/schedulers) لمعرفة كيفية استكشاف المقايضة بين سرعة وجودة الجدول، وقسم [إعادة استخدام المكونات عبر الأنابيب](../../using-diffusers/loading#reuse-components-across-pipelines) لمعرفة كيفية تحميل المكونات نفسها بكفاءة في أنابيب متعددة.

# ControlNet-XS

قدم Denis Zavadski وCarsten Rother ControlNet-XS في [ControlNet-XS](https://vislearn.github.io/ControlNet-XS/) . ويستند إلى الملاحظة التي تفيد بأن نموذج التحكم في [ControlNet الأصلي](https://huggingface.co/papers/2302.05543) يمكن أن يكون أصغر بكثير ولا يزال ينتج نتائج جيدة.

مثل نموذج ControlNet الأصلي، يمكنك توفير صورة تحكم إضافية لتشكيل عملية التوليد في Stable Diffusion والتحكم فيها. على سبيل المثال، إذا قدمت خريطة عمق، فسينشئ نموذج ControlNet صورة تحافظ على المعلومات المكانية من خريطة العمق. إنها طريقة أكثر مرونة ودقة للتحكم في عملية إنشاء الصور.

ينشئ ControlNet-XS صورًا بجودة قابلة للمقارنة مع ControlNet العادي، ولكنه أسرع بنسبة 20-25% (انظر المعيار المرجعي مع StableDiffusion-XL) ويستخدم ذاكرة أقل بنسبة 45%.

فيما يلي نظرة عامة من [صفحة المشروع](https://vislearn.github.io/ControlNet-XS/):

> مع زيادة قدرات الحوسبة، يبدو أن بنيات النماذج الحالية تتبع الاتجاه المتمثل في زيادة حجم جميع المكونات دون التحقق من ضرورة القيام بذلك. في هذا المشروع، نبحث في حجم وتصميم بنية ControlNet [Zhang et al.، 2023] للتحكم في عملية إنشاء الصور باستخدام النماذج المستندة إلى Stable Diffusion. نحن نثبت أن بنية جديدة تحتوي على 1% فقط من معلمات النموذج الأساسي تحقق نتائج متقدمة على أحدث النماذج، وهي أفضل بكثير من ControlNet من حيث درجة FID. ولهذا نسميه ControlNet-XS. نقدم الكود للتحكم في StableDiffusion-XL [Podell et al.، 2023] (النموذج B، 48 مليون معلمة) وStableDiffusion 2.1 [Rombach et al. 2022] (النموذج B، 14 مليون معلمة)، وكلها بموجب ترخيص openrail.

تمت المساهمة بهذا النموذج من قبل [UmerHA](https://twitter.com/UmerHAdil).

## StableDiffusionControlNetXSPipeline

[[autodoc]] StableDiffusionControlNetXSPipeline

- all
- __call__

## StableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput