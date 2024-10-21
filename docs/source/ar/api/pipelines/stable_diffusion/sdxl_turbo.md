# SDXL Turbo

اقترح Axel Sauer وDominik Lorenz وAndreas Blattmann وRobin Rombach في [Adversarial Diffusion Distillation](https://stability.ai/research/adversarial-diffusion-distillation) طريقة SDXL Turbo في التقطير بالانتشار الضار.

ملخص الورقة هو:

*نحن نقدم طريقة تدريب جديدة تسمى Adversarial Diffusion Distillation (ADD) والتي تقوم بفعالية بتصميم نماذج انتشار الصور واسعة النطاق في 1-4 خطوات فقط مع الحفاظ على جودة الصورة العالية. نستخدم تقطير النتيجة للاستفادة من نماذج انتشار الصور واسعة النطاق الجاهزة كإشارة تعليم بالاقتران مع خسارة الخصومة لضمان دقة الصورة العالية حتى في نظام الخطوة المنخفضة المكون من خطوة أو خطوتين من خطوات المعاينة. تظهر تحليلاتنا أن نموذجنا يتفوق بوضوح على طرق الخطوات القليلة الموجودة (GANs وLatent Consistency Models) في خطوة واحدة ويصل إلى أداء نماذج الانتشار المتقدمة (SDXL) في أربع خطوات فقط. تعد ADD أول طريقة لفتح توليف الصور في الوقت الفعلي بخطوة واحدة باستخدام نماذج الأساس.*

## نصائح

- يستخدم SDXL Turbo نفس الهندسة المعمارية [SDXL](./stable_diffusion_xl)، مما يعني أنه يحتوي على نفس واجهة برمجة التطبيقات (API). يرجى الرجوع إلى مرجع [SDXL](./stable_diffusion_xl) API لمزيد من التفاصيل.

- يجب على SDXL Turbo تعطيل مقياس التوجيه عن طريق تعيين `guidance_scale=0.0`.

- يجب على SDXL Turbo استخدام `timestep_spacing='trailing'` للمخطط واستخدام ما بين 1 و4 خطوات.

- تم تدريب SDXL Turbo لتوليد صور بحجم 512x512.

- SDXL Turbo متاح للجميع، ولكنه ليس مفتوح المصدر، مما يعني أنه قد يتعين عليك شراء ترخيص النموذج لاستخدامه في التطبيقات التجارية. تأكد من قراءة [بطاقة النموذج الرسمية](https://huggingface.co/stabilityai/sdxl-turbo) لمعرفة المزيد.

<Tip>

لمعرفة كيفية استخدام SDXL Turbo في مهام مختلفة، وكيفية تحسين الأداء، وأمثلة الاستخدام الأخرى، راجع دليل [SDXL Turbo](../../../using-diffusers/sdxl_turbo).

تفقد [Stability AI](https://huggingface.co/stabilityai) منظمة Hub للوصول إلى نقاط التحقق الرسمية للنموذج الأساسي والنموذج المحسن!

</Tip>