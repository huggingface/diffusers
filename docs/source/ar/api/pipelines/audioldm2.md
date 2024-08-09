# AudioLDM 2

تم اقتراح AudioLDM 2 في [AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) بواسطة Haohe Liu et al. يأخذ AudioLDM 2 كإدخال نص موجه ويتوقع الصوت المقابل. يمكنه توليد المؤثرات الصوتية المشروطة بالنص، والكلام البشري والموسيقى.

استلهم من [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview)، AudioLDM 2 هو نموذج انتشار خفي مشروط بالنص (LDM) يتعلم التمثيلات الصوتية المستمرة من تضمين النص. يتم استخدام نموذجين لترميز النص لحساب التضمين النصي من إدخال موجه: فرع النص من [CLAP](https://huggingface.co/docs/transformers/main/en/model_doc/clap) وترميز [Flan-T5](https://huggingface.co/docs/transformers/main/en/model_doc/flan-t5). بعد ذلك، يتم إسقاط هذه التضمينات النصية إلى مساحة تضمين مشتركة بواسطة [AudioLDM2ProjectionModel](https://huggingface.co/docs/diffusers/main/api/pipelines/audioldm2#diffusers.AudioLDM2ProjectionModel). يتم استخدام [GPT2](https://huggingface.co/docs/transformers/main/en/model_doc/gpt2) _نموذج اللغة_ للتنبؤ تلقائيًا بثمانية متجهات تضمين جديدة، مشروطة بالتضمين CLAP و Flan-T5 المسقطين. يتم استخدام متجهات التضمين المولدة وتضمينات نص Flan-T5 كشرط للاهتمام المتقاطع في LDM. تعتبر [UNet](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm2#diffusers.AudioLDM2UNet2DConditionModel) من AudioLDM 2 فريدة من نوعها من حيث أنها تأخذ **اثنين** من تضمينات الاهتمام المتقاطع، على عكس شرط الاهتمام المتقاطع الواحد، كما هو الحال في معظم نماذج LDM الأخرى.

ملخص الورقة هو كما يلي:

*على الرغم من أن توليد الصوت يشترك في خصائص مشتركة عبر أنواع مختلفة من الصوت، مثل الكلام والموسيقى والمؤثرات الصوتية، إلا أن تصميم النماذج لكل نوع يتطلب مراعاة دقيقة للأهداف والانحيازات المحددة التي قد تختلف اختلافًا كبيرًا عن تلك الأنواع الأخرى. للاقتراب أكثر من منظور موحد لتوليد الصوت، تقترح هذه الورقة إطارًا يستخدم نفس طريقة التعلم لتوليد الكلام والموسيقى والمؤثرات الصوتية. يقدم إطارنا تمثيلًا عامًا للصوت، يُطلق عليه "لغة الصوت" (LOA). يمكن ترجمة أي صوت إلى LOA بناءً على AudioMAE، وهو نموذج تعلم تمثيل مسبق خاضع للإشراف الذاتي. في عملية التوليد، نقوم بترجمة أي طرائق إلى LOA باستخدام نموذج GPT-2، ونقوم بتعلم توليد الصوت الخاضع للإشراف الذاتي باستخدام نموذج انتشار خفي مشروط بـ LOA. يجلب الإطار المقترح مزايا مثل قدرات التعلم في السياق وإمكانية إعادة استخدام نماذج AudioMAE والنماذج الخفية للانتشار الخاضعة للإشراف الذاتي. تُظهر التجارب على المعايير القياسية لتحويل النص إلى صوت، والنص إلى موسيقى، والنص إلى كلام أداءً متميزًا أو تنافسيًا مقارنة بالأساليب السابقة. يمكن العثور على الرمز الخاص بنا، والنموذج المسبق التدريب، والعرض التوضيحي على [هذا الرابط https](https://audioldm.github.io/audioldm2).*

تمت المساهمة بهذه الخطوط أنابيب من قبل [sanchit-gandhi](https://huggingface.co/sanchit-gandhi) و [Nguyễn Công Tú Anh](https://github.com/tuanh123789). يمكن العثور على قاعدة الكود الأصلية في [haoheliu/audioldm2](https://github.com/haoheliu/audioldm2).

## النصائح

### اختيار نقطة تفتيش

يأتي AudioLDM2 في ثلاثة متغيرات. تنطبق نقطتا تفتيش من هذه النقاط على المهمة العامة لتحويل النص إلى صوت. تم تدريب نقطة التفتيش الثالثة حصريًا على توليد الموسيقى النصية.

تشترك جميع نقاط التفتيش في نفس حجم النموذج لترميز النص و VAE. تختلف في حجم ودرجة UNet.

انظر الجدول أدناه للحصول على تفاصيل حول نقاط التفتيش الثلاث:

| نقطة تفتيش | المهمة | حجم نموذج UNet | حجم النموذج الإجمالي | بيانات التدريب / ساعة |
| --- | --- | --- | --- | --- |
| [audioldm2](https://huggingface.co/cvssp/audioldm2) | نص إلى صوت | 350M | 1.1B | 1150k |
| [audioldm2-large](https://huggingface.co/cvssp/audioldm2-large) | نص إلى صوت | 750M | 1.5B | 1150k |
| [audioldm2-music](https://huggingface.co/cvssp/audioldm2-music) | نص إلى موسيقى | 350M | 1.1B | 665k |
| [audioldm2-gigaspeech](https://huggingface.co/anhnct/audioldm2_gigaspeech) | نص إلى كلام | 350M | 1.1B | 10k |
| [audioldm2-ljspeech](https://huggingface.co/anhnct/audioldm2_ljspeech) | نص إلى كلام | 350M | 1.1B | |

### بناء موجه

* تعمل إدخالات الموجه الوصفية بشكل أفضل: استخدم الصفات لوصف الصوت (مثل "عالي الجودة" أو "واضح") وجعل السياق الموجه محددًا (مثل "تدفق مائي في الغابة" بدلاً من "التدفق").
* من الأفضل استخدام مصطلحات عامة مثل "قطة" أو "كلب" بدلاً من الأسماء المحددة أو الأشياء المجردة التي قد لا يكون النموذج معتادًا عليها.
* يمكن أن يؤدي استخدام موجه **سلبي** إلى تحسين جودة الموجة المولدة بشكل كبير، عن طريق توجيه التوليد بعيدًا عن المصطلحات التي تتوافق مع جودة الصوت الرديئة. جرب استخدام موجه سلبي "منخفض الجودة".

### التحكم في الاستدلال

* يمكن التحكم في _جودة_ عينة الصوت المتوقعة بواسطة وسيط `num_inference_steps`؛ حيث توفر الخطوات الأعلى جودة صوت أعلى على حساب الاستدلال البطيء.
* يمكن التحكم في _طول_ عينة الصوت المتوقعة عن طريق تغيير وسيط `audio_length_in_s`.

### تقييم الموجات الصوتية المولدة:

* يمكن أن تختلف جودة الموجات الصوتية المولدة اختلافًا كبيرًا بناءً على البذور. جرب التوليد باستخدام بذور مختلفة حتى تجد توليدًا مرضيا.
* يمكن توليد عدة موجات صوتية في مرة واحدة: قم بتعيين `num_waveforms_per_prompt` إلى قيمة أكبر من 1. سيتم إجراء التهديف التلقائي بين الموجات الصوتية المولدة ونص الموجه، وسيتم تصنيف الأصوات من الأفضل إلى الأسوأ وفقًا لذلك.

يوضح المثال التالي كيفية إنشاء موسيقى جيدة وتوليد الكلام باستخدام النصائح المذكورة أعلاه: [مثال](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm2#diffusers.AudioLDM2Pipeline.__call__.example).

<Tip>
تأكد من مراجعة دليل الجداول [](../../using-diffusers/schedulers) لمعرفة كيفية استكشاف المقايضة بين سرعة الجدول وجودته، وانظر قسم [إعادة استخدام المكونات عبر الأنابيب](../../using-diffusers/loading#reuse-components-across-pipelines) لمعرفة كيفية تحميل المكونات نفسها بكفاءة في خطوط أنابيب متعددة.
</Tip>

## AudioLDM2Pipeline

[[autodoc]] AudioLDM2Pipeline

- all
- __call__

## AudioLDM2ProjectionModel

[[autodoc]] AudioLDM2ProjectionModel

- forword

## AudioLDM2UNet2DConditionModel

[[autodoc]] AudioLDM2UNet2DConditionModel

- forword

## AudioPipelineOutput

[[autodoc]] pipelines.AudioPipelineOutput