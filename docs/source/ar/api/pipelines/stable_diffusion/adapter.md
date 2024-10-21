# T2I-Adapter

[T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.08453) بواسطة تشونغ مو، شينتاو وانغ، ليانجبين شي، جيان جانغ، زونغانغ كي، يينغ شان، شياوهو كيه.

ترجمة النص:

من خلال استخدام النماذج المُدربة مسبقًا، يمكننا توفير صور تحكم (على سبيل المثال، خريطة عمق) للتحكم في عملية توليد الصور بناءً على النص باستخدام Stable Diffusion، بحيث تتبع البنية المحددة في صورة العمق وتقوم بملء التفاصيل.

ملخص الورقة البحثية هو كما يلي:

*أظهرت القدرة التوليدية المذهلة للنماذج الضخمة للتحويل من نص إلى صورة (T2I) قوة قوية في تعلم البنى المعقدة والدلاليات ذات المعنى. ومع ذلك، فإن الاعتماد على النصوص الموجهة فقط لا يمكن أن يستفيد بشكل كامل من المعرفة التي تعلمها النموذج، خاصة عندما تكون هناك حاجة إلى التحكم المرن والدقيق (مثل اللون والبنية). في هذه الورقة، نهدف إلى "استخراج" القدرات التي تعلمتها نماذج T2I ضمنيًا، ثم استخدامها صراحة للتحكم في التوليد بشكل أكثر دقة. وعلى وجه التحديد، نقترح تعلم محولات T2I بسيطة وخفيفة الوزن لمواءمة المعرفة الداخلية في نماذج T2I مع إشارات التحكم الخارجية، مع تجميد النماذج الكبيرة الأصلية لـ T2I. بهذه الطريقة، يمكننا تدريب محولات مختلفة وفقًا لشروط مختلفة، وتحقيق التحكم والتأثيرات التحريرية الغنية في لون وبنية نتائج التوليد. علاوة على ذلك، تتمتع محولات T2I المقترحة بخصائص جذابة ذات قيمة عملية، مثل قابلية التركيب وقابلية التعميم. تُظهر التجارب المكثفة أن محول T2I الخاص بنا يتمتع بجودة توليد واعدة ومجموعة واسعة من التطبيقات.*

تمت المساهمة بهذا النموذج من قبل مساهم المجتمع [HimariO](https://github.com/HimariO) ❤️.

## StableDiffusionAdapterPipeline

[[autodoc]] StableDiffusionAdapterPipeline

- all
- __call__
- enable_attention_slicing
- disable_attention_slicing
- enable_vae_slicing
- disable_vae_slicing
- enable_xformers_memory_efficient_attention
- disable_xformers_memory_efficient_attention

## StableDiffusionXLAdapterPipeline

[[autodoc]] StableDiffusionXLAdapterPipeline

- all
- __call__
- enable_attention_slicing
- disable_attention_slicing
- enable_vae_slicing
- disable_vae_slicing
- enable_xformers_memory_efficient_attention
- disable_xformers_memory_efficient_attention