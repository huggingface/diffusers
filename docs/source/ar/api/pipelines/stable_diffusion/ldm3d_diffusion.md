# Text-to-(RGB, depth)
LDM3D هو نموذج تم اقتراحه في [LDM3D: Latent Diffusion Model for 3D](https://huggingface.co/papers/2305.10853) من قبل Gabriela Ben Melech Stan, Diana Wofk, Scottie Fox, Alex Redden, Will Saxton, Jean Yu, Estelle Aflalo, Shao-Yen Tseng, Fabio Nonato, Matthias Muller, and Vasudev Lal. يقوم LDM3D بتوليد صورة وخارطة عمق من موجه نصي معين على عكس نماذج النشر النصي-إلى-الصورة الحالية مثل [Stable Diffusion](./overview) التي تقوم فقط بتوليد صورة. وبنفس عدد المعلمات تقريبًا، ينجح LDM3D في إنشاء مساحة مضغوطة يمكن أن تحتوي على كل من صور RGB وخرائط العمق.

هناك نسختان متاحتان للاستخدام:
- [ldm3d-original](https://huggingface.co/Intel/ldm3d). نسخة الأصلية المستخدمة في [الورقة](https://arxiv.org/pdf/2305.10853.pdf)
- [ldm3d-4c](https://huggingface.co/Intel/ldm3d-4c). الإصدار الجديد من LDM3D باستخدام إدخالات 4 قنوات بدلاً من إدخالات 6 قنوات وضبط دقيق على صور عالية الدقة.

الملخص من الورقة هو:
*تقترح هذه الورقة البحثية نموذج نشر لاتنتي للـ 3D (LDM3D) يقوم بتوليد كل من بيانات الصور وخرائط العمق من موجه نصي معين، مما يسمح للمستخدمين بتوليد صور RGBD من موجهات نصية. يتم ضبط نموذج LDM3D الدقيق على مجموعة بيانات من الرباعيات التي تحتوي على صورة RGB، وخارطة عمق وتعليق، ويتم التحقق من صحتها من خلال تجارب شاملة. كما نقوم بتطوير تطبيق يسمى DepthFusion، والذي يستخدم الصور RGB المولدة وخرائط العمق لإنشاء تجارب غامرة وتفاعلية لعرض 360 درجة باستخدام TouchDesigner. تمتلك هذه التكنولوجيا القدرة على تحويل مجموعة واسعة من الصناعات، من الترفيه والألعاب إلى الهندسة المعمارية والتصميم. بشكل عام، تقدم هذه الورقة مساهمة كبيرة في مجال الذكاء الاصطناعي التوليدي والرؤية الحاسوبية، وتسلط الضوء على إمكانات LDM3D وDepthFusion لثورة إنشاء المحتوى والتجارب الرقمية. يمكن العثور على مقطع فيديو قصير يلخص النهج على [هذا الرابط](https://t.ly/tdi2).*

<Tip>
تأكد من الاطلاع على قسم [نصائح Stable Diffusion](overview#tips) لمعرفة كيفية استكشاف المقايضة بين سرعة الجدولة والجودة، وكيفية إعادة استخدام مكونات الأنابيب بكفاءة!
</Tip>

## StableDiffusionLDM3DPipeline
[[autodoc]] pipelines.stable_diffusion_ldm3d.pipeline_stable_diffusion_ldm3d.StableDiffusionLDM3DPipeline
- all
- __call__

## LDM3DPipelineOutput
[[autodoc]] pipelines.stable_diffusion_ldm3d.pipeline_stable_diffusion_ldm3d.LDM3DPipelineOutput
- all
- __call__

# Upscaler
[LDM3D-VR](https://arxiv.org/pdf/2311.03226.pdf) هو إصدار موسع من LDM3D.

ملخص الورقة هو:
*أثبتت نماذج الانتشار المخفية أنها الأفضل في إنشاء المخرجات المرئية والتلاعب بها. ومع ذلك، حسب علمنا، لا يزال إنشاء خرائط العمق بشكل مشترك مع RGB محدودًا. نقدم LDM3D-VR، وهي مجموعة من نماذج الانتشار التي تستهدف تطوير الواقع الافتراضي والتي تتضمن LDM3D-pano وLDM3D-SR. تمكن هذه النماذج من إنشاء صور بانورامية RGBD بناءً على موجهات نصية وتصغير الإدخالات منخفضة الدقة إلى صور RGBD عالية الدقة، على التوالي. يتم ضبط نماذجنا الدقيقة مسبقًا من النماذج الموجودة مسبقًا على مجموعات بيانات تحتوي على صور RGB بانورامية/عالية الدقة، وخرائط العمق والتعليقات التوضيحية. يتم تقييم كلا النموذجين بالمقارنة مع الأساليب الحالية ذات الصلة*

هناك نسختان متاحتان للاستخدام:
- [ldm3d-pano](https://huggingface.co/Intel/ldm3d-pano). تمكن هذه النسخة من إنشاء صور بانورامية وتتطلب استخدام خط أنابيب StableDiffusionLDM3D.
- [ldm3d-sr](https://huggingface.co/Intel/ldm3d-sr). تمكن هذه النسخة من تصغير صور RGB وصور العمق. يمكن استخدامه بشكل متتالي بعد خط أنابيب LDM3D الأصلي باستخدام StableDiffusionUpscaleLDM3DPipeline من خط أنابيب communauty.