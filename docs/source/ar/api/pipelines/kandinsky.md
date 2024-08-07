# Kandinsky 2.1

تم إنشاء Kandinsky 2.1 بواسطة [Arseniy Shakhmatov] (https://github.com/cene555)، [Anton Razzhigaev] (https://github.com/razzant)، [Aleksandr Nikolich] (https://github.com/AlexWortega)، [Vladimir Arkhipkin] (https://github.com/oriBetelgeuse)، [Igor Pavlov] (https://github.com/boomb0om)، [Andrey Kuznetsov] (https://github.com/kuznetsoffandrey)، و [Denis Dimitrov] (https://github.com/denndimitrov).

الوصف من صفحته على GitHub هو:

> *يتبنى Kandinsky 2.1 أفضل الممارسات من Dall-E 2 و Latent diffusion، مع تقديم بعض الأفكار الجديدة. يستخدم كنموذج مشفر للنص والصورة نموذج CLIP ومؤشر انتشار الصورة (mapping) بين المساحات الخفية لوضعيات CLIP. يزيد هذا النهج من الأداء المرئي للنموذج ويفتح آفاقًا جديدة في مزج الصور والتلاعب بالصور الموجهة بالنص.*

يمكن العثور على كود المصدر الأصلي في [ai-forever/Kandinsky-2] (https://github.com/ai-forever/Kandinsky-2).

<Tip>

اطلع على منظمة [Kandinsky Community] (https://huggingface.co/kandinsky-community) على Hub لمعرفة نقاط التحقق الرسمية للمهام مثل text-to-image و image-to-image و inpainting.

</Tip>

<Tip>

تأكد من الاطلاع على دليل Schedulers [guide] (../../using-diffusers/schedulers) لمعرفة كيفية استكشاف المقايضة بين سرعة الجدولة والجودة، وقسم [إعادة استخدام المكونات عبر الأنابيب] (../../using-diffusers/loading#reuse-components-across-pipelines) لمعرفة كيفية تحميل المكونات نفسها بكفاءة في أنابيب متعددة.

</Tip>

## KandinskyPriorPipeline

[[autodoc]] KandinskyPriorPipeline

- all
- __call__
- interpolate

## KandinskyPipeline

[[autodoc]] KandinskyPipeline

- all
- __call__

## KandinskyCombinedPipeline

[[autodoc]] KandinskyCombinedPipeline

- all
- __call__

## KandinskyImg2ImgPipeline

[[autodoc]] KandinskyImg2ImgPipeline

- all
- __call__

## KandinskyImg2ImgCombinedPipeline

[[autodoc]] KandinskyImg2ImgCombinedPipeline

- all
- __call__

## KandinskyInpaintPipeline

[[autodoc]] KandinskyInpaintPipeline

- all
- __call__

## KandinskyInpaintCombinedPipeline

[[autodoc]] KandinskyInpaintCombinedPipeline

- all
- __call__