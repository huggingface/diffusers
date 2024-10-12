لمزيد من المعلومات، راجع [Fantasy-Studio/Paint-by-Example](https://github.com/Fantasy-Studio/Paint-by-Example) و [دليل مساحات Hugging Face](https://huggingface.co/spaces).

# Paint by Example

تم تحقيق "Paint by Example: Exemplar-based Image Editing with Diffusion Models" بواسطة Binxin Yang و Shuyang Gu و Bo Zhang و Ting Zhang و Xuejin Chen و Xiaoyan Sun و Dong Chen و Fang Wen.

ملخص الورقة هو:

> حقق تحرير الصور الموجهة باللغة نجاحًا كبيرًا في الآونة الأخيرة. وفي هذه الورقة، ولأول مرة، نبحث في تحرير الصور الموجهة بالأمثلة للتحكم بشكل أكثر دقة. نحن نحقق هذا الهدف من خلال الاستفادة من التدريب الذاتي الإشراف لفصل وإعادة تنظيم صورة المصدر والمثال. ومع ذلك، فإن النهج البسيط سيؤدي إلى آثار اندماج واضحة. نقوم بتحليله بعناية ونقترح عنق زجاجة المعلومات وتعزيزات قوية لتجنب الحل البديهي المتمثل في نسخ ولصق صورة المثال مباشرة. وفي الوقت نفسه، لضمان قابلية التحكم في عملية التحرير، نقوم بتصميم قناع ذي شكل تعسفي لصورة المثال ونستفيد من الإرشادات الخالية من التصنيف لزيادة التشابه مع صورة المثال. يتضمن الإطار الكامل تمريرًا واحدًا لنموذج الانتشار دون أي تحسين تكراري. نثبت أن طريقة عملنا تحقق أداءً رائعًا وتمكن من تحرير الصور الواقعية مع الحفاظ على الإخلاص العالي.

يمكن العثور على قاعدة الكود الأصلية في [Fantasy-Studio/Paint-by-Example](https://github.com/Fantasy-Studio/Paint-by-Example)، ويمكنك تجربتها في [هذا العرض التوضيحي](https://huggingface.co/spaces/Fantasy-Studio/Paint-by-Example).

## نصائح

يتم دعم Paint by Example بواسطة نقطة التفتيش الرسمية [Fantasy-Studio/Paint-by-Example](https://huggingface.co/Fantasy-Studio/Paint-by-Example). تم تسخين نقطة التفتيش مسبقًا من [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) لملء صور الأقنعة جزئيًا بشرط وجود صور مثال ومرجع.

<Tip>

تأكد من مراجعة دليل Schedulers [guide](../../using-diffusers/schedulers) لمعرفة كيفية استكشاف المقايضة بين سرعة الجدولة والجودة، وقسم [إعادة استخدام المكونات عبر الأنابيب](../../using-diffusers/loading#reuse-components-across-pipelines) لمعرفة كيفية تحميل المكونات نفسها بكفاءة في أنابيب متعددة.

</Tip>

## PaintByExamplePipeline

[[autodoc]] PaintByExamplePipeline

- all

- __call__

## StableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput