# Kandinsky 2.2

تم إنشاء Kandinsky 2.2 بواسطة [Arseniy Shakhmatov] (https://github.com/cene555)، [Anton Razzhigaev] (https://github.com/razzant)، [Aleksandr Nikolich] (https://github.com/AlexWortega)، [Vladimir Arkhipkin] (https://github .com/oriBetelgeuse)، [Igor Pavlov] (https://github.com/boomb0om)، [Andrey Kuznetsov] (https://github.com/kuznetsoffandrey)، و [Denis Dimitrov] (https://github.com/denndimitrov).

الوصف من صفحته على GitHub هو:

*يأتي Kandinsky 2.2 بتحسينات كبيرة مقارنة بالإصدار السابق، Kandinsky 2.1، من خلال تقديم ترميز صورة جديد وأكثر قوة - CLIP-ViT-G ودعم ControlNet. يؤدي التبديل إلى CLIP-ViT-G كترميز للصورة إلى زيادة كبيرة في قدرة النموذج على إنشاء صور جمالية أكثر وفهم النص بشكل أفضل، مما يعزز الأداء العام للنموذج. تسمح إضافة آلية ControlNet للنموذج بالتحكم الفعال في عملية إنشاء الصور. يؤدي هذا إلى مخرجات أكثر دقة وجمالية من الناحية المرئية ويفتح إمكانيات جديدة للتلاعب بالصور الموجهة بالنص.*

يمكن العثور على الكود الأصلي في [ai-forever/Kandinsky-2] (https://github.com/ai-forever/Kandinsky-2).

<Tip>

اطلع على منظمة [Kandinsky Community] (https://huggingface.co/kandinsky-community) على Hub لمعرفة نقاط تفتيش النموذج الرسمية لمهام مثل النص إلى الصورة والصورة إلى الصورة و inpainting.

</Tip>

<Tip>

تأكد من مراجعة دليل الجداول الزمنية [guide] (../../ using-diffusers / schedulers) لمعرفة كيفية استكشاف المقايضة بين سرعة وجودة الجدول الزمني، وقسم [إعادة استخدام المكونات عبر الأنابيب] (../../ using-diffusers / loading # إعادة استخدام المكونات عبر الأنابيب) لمعرفة كيفية تحميل المكونات نفسها بكفاءة في أنابيب متعددة.

</Tip>

## KandinskyV22PriorPipeline

[[autodoc]] KandinskyV22PriorPipeline

- all
- __call__
- استيفاء

## KandinskyV22Pipeline

[[autodoc]] KandinskyV22Pipeline

- all
- __call__

## KandinskyV22CombinedPipeline

[[autodoc]] KandinskyV22CombinedPipeline

- all
- __call__

## KandinskyV22ControlnetPipeline

[[autodoc]] KandinskyV22ControlnetPipeline

- all
- __call__

## KandinskyV22PriorEmb2EmbPipeline

[[autodoc]] KandinskyV22PriorEmb2EmbPipeline

- all
- __call__
- استيفاء

## KandinskyV22Img2ImgPipeline

[[autodoc]] KandinskyV22Img2ImgPipeline

- all
- __call__

## KandinskyV22Img2ImgCombinedPipeline

[[autodoc]] KandinskyV22Img2ImgCombinedPipeline

- all
- __call__

## KandinskyV22ControlnetImg2ImgPipeline

[[autodoc]] KandinskyV22ControlnetImg2ImgPipeline

- all
- __call__

## KandinskyV22InpaintPipeline

[[autodoc]] KandinskyV22InpaintPipeline

- all
- __call__

## KandinskyV22InpaintCombinedPipeline

[[autodoc]] KandinskyV22InpaintCombinedPipeline

- all
- __call__