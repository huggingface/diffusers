# LEDITS++

تم اقتراح LEDITS++ في [LEDITS++: Limitless Image Editing using Text-to-Image Models](https://huggingface.co/papers/2311.16711) بواسطة Manuel Brack، و Felix Friedrich، و Katharina Kornmeier، و Linoy Tsaban، و Patrick Schramowski، و Kristian Kersting، و Apolinário Passos.

ملخص الورقة البحثية هو:

*حظيت نماذج النص إلى الصورة القائمة على التشتت مؤخرًا باهتمام متزايد لقدرتها المذهلة على إنتاج صور عالية الدقة من النصوص فقط. وتهدف جهود البحث اللاحقة إلى استغلال وتطبيق قدراتها على تحرير الصور الحقيقية. ومع ذلك، فإن الطرق الحالية للتحويل من صورة إلى صورة غالباً ما تكون غير فعالة وغير دقيقة ومحدودة المرونة. فهي إما تتطلب ضبطًا دقيقًا يستغرق وقتًا طويلاً، أو تنحرف بشكل غير ضروري عن الصورة الأصلية، و/أو تفتقر إلى الدعم لإجراء تعديلات متعددة في نفس الوقت. ولمعالجة هذه القضايا، نقدم LEDITS++، وهي تقنية فعالة ورغم ذلك متعددة الاستخدامات ودقيقة للتلاعب بالصور النصية. لا يتطلب نهج الانعكاس المبتكر في LEDITS++ أي ضبط أو تحسين وينتج نتائج عالية الدقة في بضع خطوات من التشتت. ثانيًا، تدعم منهجيتنا إجراء تعديلات متعددة في نفس الوقت وهي مستقلة عن البنية. ثالثًا، نستخدم تقنية قناع ضمني جديدة تحد من التغييرات في مناطق الصورة ذات الصلة. نقترح معيار TEdBench++ الجديد كجزء من تقييمنا الشامل. توضح نتائجنا قدرات LEDITS++ وتحسيناتها مقارنة بالطرق السابقة. صفحة المشروع متاحة على https://leditsplusplus-project.static.hf.space.*

<Tip>
يمكنك العثور على معلومات إضافية حول LEDITS++ على [صفحة المشروع](https://leditsplusplus-project.static.hf.space/index.html) وتجربتها في [العرض التوضيحي](https://huggingface.co/spaces/editing-images/leditsplusplus).
</Tip>

<Tip warning={true}>
بسبب بعض مشكلات التوافق مع الإصدارات السابقة مع التنفيذ الحالي لـ [`~schedulers.DPMSolverMultistepScheduler`] في برامج التشتت، لم يعد بإمكان هذا التنفيذ لـ LEdits++ ضمان الانعكاس المثالي.
من غير المرجح أن يكون لهذه المشكلة أي تأثير ملحوظ على حالات الاستخدام التطبيقية. ومع ذلك، نقدم تنفيذًا بديلاً يضمن الانعكاس المثالي في مستودع مخصص على GitHub](https://github.com/ml-research/ledits_pp).
</Tip>

نقدم خطي إنتاج متميزين بناءً على نماذج مسبقة التدريب المختلفة.

## LEditsPPPipelineStableDiffusion

[[autodoc]] pipelines.ledits_pp.LEditsPPPipelineStableDiffusion
- all
- __call__
- invert

## LEditsPPPipelineStableDiffusionXL

[[autodoc]] pipelines.ledits_pp.LEditsPPPipelineStableDiffusionXL
- all
- __call__
- invert

## LEditsPPDiffusionPipelineOutput

[[autodoc]] pipelines.ledits_pp.pipeline_output.LEditsPPDiffusionPipelineOutput
- all

## LEditsPPInversionPipelineOutput

[[autodoc]] pipelines.ledits_pp.pipeline_output.LEditsPPInversionPipelineOutput
- all