لمزيد من المعلومات، راجع [رخصة أباتشي الإصدار 2.0](http://www.apache.org/licenses/LICENSE-2.0).

# aMUSEd

تم تقديم aMUSEd في [aMUSEd: An Open MUSE Reproduction](https://huggingface.co/papers/2401.01808) بواسطة سوراج باتيل، ويليام بيرمان، روبن رومباخ، وباتريك فون بلاتين.

aMUSEd هو نموذج خفيف الوزن للتحويل من نص إلى صورة يعتمد على بنية MUSE. وهو مفيد بشكل خاص في التطبيقات التي تتطلب نموذجًا خفيفًا وسريعًا، مثل إنشاء العديد من الصور بسرعة في نفس الوقت.

aMUSEd هو محول قائم على رموز VQVAE يمكنه إنشاء صورة في عدد أقل من عمليات التمرير الأمامي مقارنة بالعديد من نماذج الانتشار. وعلى عكس MUSE، فإنه يستخدم مشفر النص الأصغر CLIP-L/14 بدلاً من t5-xxl. وبفضل عدد معلماتها الصغير وعملية التوليد التي تتطلب عددًا قليلاً من عمليات التمرير الأمامي، يمكن لـ aMUSEd إنشاء العديد من الصور بسرعة. وتظهر هذه الميزة بشكل خاص عند زيادة حجم الدفعات.

الملخص من الورقة هو:

*نقدم aMUSEd، وهو نموذج مفتوح المصدر وخفيف الوزن للصور المقنعة (MIM) لتوليد الصور بناءً على النص باستخدام MUSE. مع 10% من معلمات MUSE، يركز aMUSEd على سرعة إنشاء الصور. نعتقد أن MIM لم يتم استكشافه بشكل كاف مقارنة بالانتشار الكامن، وهو النهج السائد لتوليد الصور بناءً على النص. مقارنة بالانتشار الكامن، يتطلب MIM خطوات استنتاج أقل وهو أكثر قابلية للفهم. بالإضافة إلى ذلك، يمكن ضبط MIM دقيقًا لتعلم أنماط إضافية بصورة واحدة فقط. نأمل في تشجيع المزيد من الاستكشاف لـ MIM من خلال توضيح فعاليته في توليد الصور واسعة النطاق بناءً على النص وإصدار شفرة التدريب القابلة للتكرار. كما نقوم بإصدار نقاط تفتيش لنموذجين ينتجان صورًا مباشرة بدقة 256x256 و512x512.*

| النموذج | المعلمات |
| ------ | -------- |
| [amused-256](https://huggingface.co/amused/amused-256) | 603M |
| [amused-512](https://huggingface.co/amused/amused-512) | 608M |

## AmusedPipeline

[[autodoc]] AmusedPipeline
- __call__
- all
- enable_xformers_memory_efficient_attention
- disable_xformers_memory_efficient_attention

[[autodoc]] AmusedImg2ImgPipeline
- __call__
- all
- enable_xformers_memory_efficient_attention
- disable_xformers_memory_efficient_attention

[[autodoc]] AmusedInpaintPipeline
- __call__
- all
- enable_xformers_memory_efficient_attention
- disable_xformers_memory_efficient_attention