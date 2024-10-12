# Kandinsky 3

تم إنشاء Kandinsky 3 بواسطة [فلاديمير أركيبكين](https://github.com/oriBetelgeuse)، [أناستازيا مالتسيفا](https://github.com/NastyaMittseva)، [إيجور بافلوف](https://github.com/boomb0om)، [أندري فيلاتوف](https://github.com/anvilarth)، [أرسيني شاخماتوف](https://github.com/cene555)، [أندريه كوزنيتسوف](https://github.com/kuznetsoffandrey)، [دينيس ديميتروف](https://github.com/denndimitrov)، [زين شاهين](https://github.com/zeinsh)

الوصف من صفحته على GitHub:

*Kandinsky 3.0* هو نموذج مفتوح المصدر للتحويل من نص إلى صورة مبني على عائلة النماذج Kandinsky2-x. وبالمقارنة مع الإصدارات السابقة، تم إجراء تحسينات على فهم النص والجودة المرئية للنموذج، والتي تم تحقيقها من خلال زيادة حجم مشفر النص ونماذج Diffusion U-Net، على التوالي.

يتضمن تصميمه 3 مكونات رئيسية:

1. [FLAN-UL2](https://huggingface.co/google/flan-ul2)، وهو نموذج ترميز وفك ترميز يعتمد على تصميم T5.
2. تصميم جديد لـ U-Net مع كتل BigGAN-deep، والذي يضاعف العمق مع الحفاظ على نفس عدد المعلمات.
3. Sber-MoVQGAN هو فك تشفير أثبت تفوقه في استعادة الصور.

يمكن العثور على كود المصدر الأصلي في [ai-forever/Kandinsky-3](https://github.com/ai-forever/Kandinsky-3).

<Tip>

اطلع على منظمة [Kandinsky Community](https://huggingface.co/kandinsky-community) على Hub للحصول على نقاط تفتيش النموذج الرسمية لمهام مثل التحويل من نص إلى صورة، ومن صورة إلى صورة، والطلاء.

</Tip>

<Tip>

تأكد من مراجعة دليل الجداول الزمنية [guide](../../using-diffusers/schedulers) لمعرفة كيفية استكشاف المقايضة بين سرعة وجودة الجدول الزمني، وقسم [إعادة استخدام المكونات عبر الأنابيب](../../using-diffusers/loading#reuse-components-across-pipelines) لمعرفة كيفية تحميل المكونات نفسها بكفاءة في أنابيب متعددة.

</Tip>

## Kandinsky3Pipeline

[[autodoc]] Kandinsky3Pipeline
- all
- __call__

## Kandinsky3Img2ImgPipeline

[[autodoc]] Kandinsky3Img2ImgPipeline
- all
- __call__