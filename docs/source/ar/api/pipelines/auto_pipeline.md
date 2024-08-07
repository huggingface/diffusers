# AutoPipeline

صُممت `AutoPipeline` لتسهيل تحميل نقطة تفتيش لمهمة دون الحاجة إلى معرفة فئة الأنابيب المحددة. بناءً على المهمة، تقوم `AutoPipeline` باسترداد فئة الأنابيب الصحيحة تلقائيًا من ملف `model_index.json` الخاص بنقطة التفتيش.

> [!TIP]
> اطلع على [دروس AutoPipeline](../../tutorials/autopipeline) لمعرفة كيفية استخدام هذا الـ API!

## AutoPipelineForText2Image

[[autodoc]] AutoPipelineForText2Image

- all
- from_pretrained
- from_pipe

## AutoPipelineForImage2Image

[[autodoc]] AutoPipelineForImage2Image

- all
- from_pretrained
- from_pipe

## AutoPipelineForInpainting

[[autodoc]] AutoPipelineForInpainting

- all
- from_pretrained
- from_pipe