# InstructPix2Pix

[InstructPix2Pix: Learning to Follow Image Editing Instructions](https://huggingface.co/papers/2211.09800) هو من تأليف Tim Brooks وAleksander Holynski وAlexei A. Efros.

الملخص المستخرج من الورقة هو:

*نقترح طريقة لتحرير الصور بناءً على تعليمات بشرية: بالنظر إلى صورة دخل وتعليمة مكتوبة تخبر النموذج بما يجب فعله، يتبع نموذجنا هذه التعليمات لتحرير الصورة. وللحصول على بيانات التدريب لهذه المشكلة، نجمع معرفة نموذجين كبيرين مُدربين مسبقًا - نموذج لغوي (GPT-3) ونموذج نص-إلى-صورة (Stable Diffusion) - لإنشاء مجموعة بيانات كبيرة من أمثلة تحرير الصور. تم تدريب نموذجنا، InstructPix2Pix، المشروط بالانتشار، على البيانات التي تم إنشاؤها، ويقوم بالتعميم على الصور الحقيقية وتعليمات المستخدم المكتوبة أثناء الاستدلال. نظرًا لأنه يقوم بالتحرير في الاتجاه الأمامي ولا يتطلب ضبطًا دقيقًا أو عكسًا لكل مثال، فإن نموذجنا يقوم بتحرير الصور بسرعة، في غضون ثوانٍ. ونعرض نتائج تحرير مقنعة لمجموعة متنوعة من صور الدخل والتعليمات المكتوبة.*

يمكنك العثور على معلومات إضافية حول InstructPix2Pix على [صفحة المشروع](https://www.timothybrooks.com/instruct-pix2pix)، و [رمز المصدر الأصلي](https://github.com/timothybrooks/instruct-pix2pix)، وجربه في [العرض التوضيحي](https://huggingface.co/spaces/timbrooks/instruct-pix2pix).

<Tip>

تأكد من الاطلاع على دليل Schedulers [guide](../../using-diffusers/schedulers) لمعرفة كيفية استكشاف المقايضة بين سرعة المجدول والجودة، وقسم [إعادة استخدام المكونات عبر الأنابيب](../../using-diffusers/loading#reuse-components-across-pipelines) لمعرفة كيفية تحميل المكونات نفسها بكفاءة في أنابيب متعددة.

</Tip>

## StableDiffusionInstructPix2PixPipeline

[[autodoc]] StableDiffusionInstructPix2PixPipeline

- __call__
- all
- load_textual_inversion
- load_lora_weights
- save_lora_weights

## StableDiffusionXLInstructPix2PixPipeline

[[autodoc]] StableDiffusionXLInstructPix2PixPipeline

- __call__
- all