# BLIP-Diffusion

تم اقتراح BLIP-Diffusion في [BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing](https://arxiv.org/abs/2305.14720). فهو يمكّن الإنشاء الموجه بالموضوع بدون تدريب مسبق والتحكم الموجه بدون تدريب مسبق.

ملخص الورقة هو:

*تُنشئ نماذج توليد الصور الموجهة بالنص تجسيدات جديدة لموضوع الإدخال بناءً على مطالبات النص. تعاني النماذج الحالية من ضبط دقيق مطول وصعوبات في الحفاظ على دقة الموضوع. للتغلب على هذه القيود، نقدم BLIP-Diffusion، وهو نموذج توليد صور جديد موجه بالموضوع يدعم التحكم متعدد الوسائط والذي يستهلك إدخالات صور الموضوع ومطالبات النص. على عكس النماذج الأخرى الموجهة بالموضوع، يقدم BLIP-Diffusion مشفرًا متعدد الوسائط جديدًا يتم تدريبه مسبقًا لتوفير تمثيل الموضوع. أولاً، نقوم بتدريب المشفر متعدد الوسائط مسبقًا باتباع BLIP-2 لإنتاج تمثيل مرئي متوافق مع النص. بعد ذلك، نقوم بتصميم مهمة تعلم تمثيل الموضوع والتي تمكن نموذج الانتشار من الاستفادة من هذا التمثيل المرئي وإنشاء تجسيدات موضوع جديدة. مقارنة بالطرق السابقة مثل DreamBooth، يمكّن نموذجنا التوليد الموجه بالموضوع بدون تدريب مسبق، وضبطًا دقيقًا فعالاً لموضوع مخصص بسرعة تصل إلى 20x. كما نوضح أن BLIP-Diffusion يمكن دمجه بمرونة مع التقنيات الموجودة مثل ControlNet وprompt-to-prompt لتمكين تطبيقات التوليد والتحرير الموجهة بالموضوع الجديدة. صفحة المشروع على [هذا https URL](https://dxli94.github.io/BLIP-Diffusion-website/).*

يمكن العثور على قاعدة الكود الأصلية في [salesforce/LAVIS](https://github.com/salesforce/LAVIS/tree/main/projects/blip-diffusion). يمكنك العثور على نقاط التحقق الرسمية لـ BLIP-Diffusion تحت منظمة [hf.co/SalesForce](https://hf.co/SalesForce).

تم المساهمة بـ `BlipDiffusionPipeline` و`BlipDiffusionControlNetPipeline` بواسطة [`ayushtues`](https://github.com/ayushtues/).

<Tip>
تأكد من الاطلاع على دليل Schedulers [guide](../../using-diffusers/schedulers) لمعرفة كيفية استكشاف المقايضة بين سرعة وجودة الجدول، وانظر قسم [إعادة استخدام المكونات عبر الأنابيب](../../using-diffusers/loading#reuse-components-across-pipelines) لمعرفة كيفية تحميل المكونات نفسها بكفاءة في أنابيب متعددة.
</Tip>

## BlipDiffusionPipeline

[[autodoc]] BlipDiffusionPipeline
- all
- __call__

## BlipDiffusionControlNetPipeline

[[autodoc]] BlipDiffusionControlNetPipeline
- all
- __call__