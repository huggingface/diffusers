لمزيد من المعلومات حول MultiDiffusion، راجع [صفحة المشروع](https://multidiffusion.github.io/) و [رمز المصدر الأصلي](https://github.com/omerbt/MultiDiffusion)، وجربه في [العرض التوضيحي](https://huggingface.co/spaces/weizmannscience/MultiDiffusion).

## نصائح

عند استدعاء [`StableDiffusionPanoramaPipeline`]، يمكن تحديد `view_batch_size` بحيث يكون > 1.
بالنسبة لبعض وحدات معالجة الرسومات (GPU) ذات الأداء العالي، يمكن أن يؤدي ذلك إلى تسريع عملية التوليد وزيادة استخدام VRAM.

لإنشاء صور مشابهة لصور panorama، تأكد من تمرير معلمة العرض بشكل مناسب. نوصي بقيمة عرض تبلغ 2048 وهي القيمة الافتراضية.

يتم تطبيق الحشو الدائري لضمان عدم وجود أي آثار للترقيع عند العمل مع صور panorama، مما يضمن انتقالًا سلسًا من الجزء الأيمن إلى الجزء الأيسر. من خلال تمكين الحشو الدائري (ضبط `circular_padding=True`)، تطبق العملية اقتصاصات إضافية بعد النقطة الأبعد إلى اليمين في الصورة، مما يسمح للنموذج "برؤية" الانتقال من الجزء الأيمن إلى الجزء الأيسر. يساعد ذلك في الحفاظ على الاتساق البصري بزاوية 360 درجة، ويخلق "panorama" صحيحًا يمكن عرضه باستخدام عارضات panorama بزاوية 360 درجة. عند فك تشفير latents في Stable Diffusion، يتم تطبيق الحشو الدائري لضمان مطابقة latents المفككة في مساحة RGB.

على سبيل المثال، بدون الحشو الدائري، هناك أثر للترقيع (القيمة الافتراضية):

![img](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/indoor_%20no_circular_padding.png)

ولكن مع الحشو الدائري، يتطابق الجزء الأيمن والأيسر (`circular_padding=True`):

![img](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/indoor_%20circular_padding.png)

<Tip>

تأكد من الاطلاع على دليل Schedulers [guide](../../using-diffusers/schedulers) لمعرفة كيفية استكشاف المقايضة بين سرعة الجدول الزمني والجودة، وقسم [إعادة استخدام المكونات عبر الأنابيب](../../using-diffusers/loading#reuse-components-across-pipelines) لمعرفة كيفية تحميل المكونات نفسها بكفاءة في أنابيب متعددة.

</Tip>

## StableDiffusionPanoramaPipeline

[[autodoc]] StableDiffusionPanoramaPipeline

- __call__

- all

## StableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput