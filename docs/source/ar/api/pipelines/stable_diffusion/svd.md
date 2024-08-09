# Stable Video Diffusion

تم اقتراح Stable Video Diffusion في الورقة البحثية [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://hf.co/papers/2311.15127) بواسطة أندرياس بلاتمان، وتيم دوكهورن، وسوميث كولال، ودانييل مينديليفيتش، وماتشي كيليان، ودومينيك لورنز، ويام ليفي، وزيون إنجلش، وفيكرام فوليتي، وآدم ليتس، وفارون جامباني، وروبين رومباخ.

ملخص الورقة البحثية هو:

*نحن نقدم Stable Video Diffusion - وهو نموذج انتشار فيديو كامن لتقنية text-to-video و image-to-video بجودة عالية. مؤخرًا، تم تحويل نماذج الانتشار الكامنة المدربة على توليد الصور ثنائية الأبعاد إلى نماذج فيديو مولدة عن طريق إدراج طبقات زمنية وضبط دقيق لها على مجموعات بيانات الفيديو الصغيرة والعالية الجودة. ومع ذلك، تختلف طرق التدريب في الأدبيات اختلافًا كبيرًا، ولم يتفق المجال بعد على استراتيجية موحدة لتنظيم بيانات الفيديو. في هذه الورقة، نقوم بتحديد وتقييم ثلاث مراحل مختلفة للتدريب الناجح لنماذج LDM للفيديو: pretraining text-to-image، وvideo pretraining، وhigh-quality video finetuning. علاوة على ذلك، نثبت ضرورة وجود مجموعة بيانات تدريب مُدارة جيدًا لتوليد مقاطع فيديو عالية الجودة ونقدم عملية تنظيم منهجية لتدريب نموذج أساسي قوي، بما في ذلك استراتيجيات التعليق والتصفية. بعد ذلك، نستكشف تأثير الضبط الدقيق لنموذجنا الأساسي على بيانات عالية الجودة ونقوم بتدريب نموذج text-to-video تنافسي مع تقنية video generation مغلقة المصدر. كما نوضح أن نموذجنا الأساسي يوفر تمثيل حركة قوي للمهام اللاحقة مثل image-to-video generation والقدرة على التكيف مع وحدات LoRA الخاصة بحركة الكاميرا. وأخيرًا، نثبت أن نموذجنا يوفر أولوية 3D متعددة المشاهدات ويمكن أن يكون بمثابة قاعدة لضبط نموذج انتشار متعدد المشاهدات دقيق ينشئ بشكل مشترك عدة وجهات نظر للأشياء بطريقة feedforward، متفوقًا على الطرق القائمة على الصور بكسر ميزانية الحوسبة الخاصة بها.*

<Tip>

لمعرفة كيفية استخدام Stable Video Diffusion، اطلع على الدليل [Stable Video Diffusion](../../../using-diffusers/svd).

<br>

تفقد منظمة [Stability AI](https://huggingface.co/stabilityai) Hub للحصول على نقاط التحقق [base](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) و [extended frame](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)

</Tip>

## نصائح

تعد تقنية video generation كثيفة الاستخدام للذاكرة، وأحد الطرق لتقليل استخدام الذاكرة هي تعيين `enable_forward_chunking` على شبكة UNet الخاصة بالخط الأنابيب بحيث لا يتم تشغيل طبقة التغذية الأمامية بالكامل مرة واحدة. من الأكثر كفاءة تقسيمها إلى مجموعات في حلقة.

راجع الدليل [Text or image-to-video](text-img2vid) لمزيد من التفاصيل حول كيفية تأثير بعض المعلمات على إنشاء الفيديو وكيفية تحسين الاستدلال عن طريق تقليل استخدام الذاكرة.

## StableVideoDiffusionPipeline

[[autodoc]] StableVideoDiffusionPipeline

## StableVideoDiffusionPipelineOutput

[[autodoc]] pipelines.stable_video_diffusion.StableVideoDiffusionPipelineOutput