# أنابيب Marigold لمهام رؤية الكمبيوتر

اقترحت Marigold في [إعادة استخدام مولدات الصور القائمة على الانتشار لتقدير العمق أحادي العين](https://huggingface.co/papers/2312.02145)، وهي ورقة شفوية CVPR 2024 بواسطة [Bingxin Ke](http://www.kebingxin.com/)، [Anton Obukhov](https://www.obukhov.ai/)، [Shengyu Huang](https://shengyuh.github.io/)، [Nando Metzger](https://nandometzger.github.io/)، [Rodrigo Caye Daudt](https://rcdaudt.github.io/)، و [Konrad Schindler](https://scholar.google.com/citations؟user=FZuNgqIAAAAJ&hl=en).

الفكرة هي إعادة استخدام الأولوية الغنية للنموذج الانتشاري القائم على النص (LDMs) لمهام رؤية الكمبيوتر التقليدية. في البداية، تم استكشاف هذه الفكرة لضبط Diffusion المستقر لتقدير العمق أحادي العين، كما هو موضح في التشويق أعلاه. في وقت لاحق،

- درب [Tianfu Wang](https://tianfwang.github.io/) أول نموذج اتساق خفي (LCM) لـ Marigold، والذي فتح الاستدلال أحادي الخطوة السريع.
- قام [Kevin Qu](https://www.linkedin.com/in/kevin-qu-b3417621b/?locale=en_US) بتوسيع النهج لتقدير القواعد السطحية.
- ساهم [Anton Obukhov](https://www.obukhov.ai/) بأنابيب والتوثيق في أجهزة الانتشار (ممكن ومدعوم من [YiYi Xu](https://yiyixuxu.github.io/) و [Sayak Paul](https://sayak.dev/)).

المستخلص من الورقة هو:

> "تقدير العمق أحادي العين هو مهمة أساسية لرؤية الكمبيوتر. إن استعادة العمق ثلاثي الأبعاد من صورة واحدة غير محددة هندسيًا وتتطلب فهم المشهد، لذلك ليس من المستغرب أن يؤدي ظهور التعلم العميق إلى تقدم كبير. يعكس التقدم المثير للإعجاب في مقدرات العمق أحادية العين نمو سعة النموذج، من شبكات CNN المتواضعة نسبيًا إلى هندسات المحول الكبيرة. ومع ذلك، تميل مقدرات العمق أحادية العين إلى النضال عندما يتم تقديمها بصور ذات محتوى وتخطيط غير مألوفين، نظرًا لأن معرفتها بالعالم المرئي مقيدة بالبيانات التي شوهدت أثناء التدريب، ويتم تحديها بالتعميم الصفري على المجالات الجديدة. وهذا يحفزنا على استكشاف ما إذا كانت الأولويات الواسعة التي تم التقاطها في نماذج الانتشار التوليدية الحديثة يمكن أن تمكن من تقدير العمق بشكل أفضل وأكثر قابلية للتعميم. نقدم Marigold، وهي طريقة لتقدير العمق أحادي العين الدافعي المشتق من Stable Diffusion والذي يحتفظ بمعرفته الأولية الغنية. يمكن ضبط المقدر في غضون بضعة أيام على وحدة معالجة رسومات واحدة باستخدام بيانات تدريب اصطناعية فقط. إنه يحقق أداءً متميزًا عبر مجموعة واسعة من مجموعات البيانات، بما في ذلك مكاسب في الأداء تزيد عن 20% في حالات محددة. صفحة المشروع: https://marigoldmonodepth.github.io. "

## الأنابيب المتاحة

يدعم كل خط أنابيب مهمة رؤية حاسوبية واحدة، والتي تأخذ كإدخال صورة RGB وتنتج *توقع* الطريقة التي تهمك، مثل خريطة العمق لصورة الإدخال.

في الوقت الحالي، يتم تنفيذ المهام التالية:

| خط الأنابيب | الطرائق المتوقعة | العروض التوضيحية |
|-------------|------------------|------------------|
| [MarigoldDepthPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/marigold/pipeline_marigold_depth.py) | [العمق](https://en.wikipedia.org/wiki/Depth_map)، [التباين](https://en.wikipedia.org/wiki/Binocular_disparity) | [العرض التوضيحي السريع (LCM)](https://huggingface.co/spaces/prs-eth/marigold-lcm)، [العرض التوضيحي الأصلي البطيء (DDIM)](https://huggingface.co/spaces/prs-eth/marigold) |
| [MarigoldNormalsPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/marigold/pipeline_marigold_normals.py) | [القواعد السطحية](https://en.wikipedia.org/wiki/Normal_mapping) | [العرض التوضيحي السريع (LCM)](https://huggingface.co/spaces/prs-eth/marigold-normals-lcm) |

## نقاط التفتيش المتاحة

يمكن العثور على نقاط التفتيش الأصلية في منظمة [PRS-ETH](https://huggingface.co/prs-eth/) على Hugging Face.

<Tip>

تأكد من مراجعة دليل [المخططين](../../using-diffusers/schedulers) لمعرفة كيفية استكشاف المقايضة بين سرعة المخطط والنوعية، وراجع قسم [إعادة استخدام المكونات عبر خطوط الأنابيب](../../using-diffusers/loading#reuse-components-across-pipelines) لمعرفة كيفية تحميل المكونات نفسها بكفاءة في خطوط أنابيب متعددة. أيضًا، لمعرفة المزيد حول تقليل استخدام الذاكرة لخط الأنابيب هذا، راجع قسم ["تقليل استخدام الذاكرة"] [هنا](../../using-diffusers/svd#reduce-memory-usage).

</Tip>

<Tip warning={true}>

تم تصميم خطوط أنابيب Marigold واختبارها فقط باستخدام `DDIMScheduler` و `LCMScheduler`. اعتمادًا على المخطط، يختلف عدد خطوات الاستدلال المطلوبة للحصول على تنبؤات موثوقة، ولا توجد قيمة عالمية تعمل بشكل أفضل عبر المخططات. لهذا السبب، يتم تعيين القيمة الافتراضية لـ `num_inference_steps` في طريقة `__call__` لخط الأنابيب إلى `None` (راجع مرجع API). ما لم يتم تعيينه بشكل صريح، فستتم استعادته من تكوين نقطة التفتيش `model_index.json`. يتم ذلك لضمان تنبؤات عالية الجودة عند استدعاء خط الأنابيب باستخدام حجة "الصورة" فقط.

</Tip>

انظر أيضًا أمثلة الاستخدام [Marigold](marigold_usage).

## MarigoldDepthPipeline

[[autodoc]] MarigoldDepthPipeline

- all
- __call__

## MarigoldNormalsPipeline

[[autodoc]] MarigoldNormalsPipeline

- all
- __call__

## MarigoldDepthOutput

[[autodoc]] pipelines.marigold.pipeline_marigold_depth.MarigoldDepthOutput

## MarigoldNormalsOutput

[[autodoc]] pipelines.marigold.pipeline_marigold_normals.MarigoldNormalsOutput