لمزيد من المعلومات حول كيفية استخدام هذا النموذج، يرجى الاطلاع على [صفحة التعليمات](https://huggingface.co/docs/hub/troubleshooting).

# نماذج الاتساق الكامنة Latent Consistency Models

اقترحت نماذج الاتساق الكامنة (LCMs) في [نماذج الاتساق الكامنة: توليف الصور عالية الدقة باستخدام الاستدلال متعدد الخطوات](https://huggingface.co/papers/2310.04378) بواسطة Simian Luo و Yiqin Tan و Longbo Huang و Jian Li و Hang Zhao.

ملخص الورقة هو كما يلي:

> "حققت نماذج الانتشار الكامنة (LDMs) نتائج ملحوظة في توليف الصور عالية الدقة. ومع ذلك، فإن عملية أخذ العينات التكرارية كثيفة الحسابات وتؤدي إلى بطء التوليد. استلهاما من نماذج الاتساق (Song et al.)، نقترح نماذج الاتساق الكامنة (LCMs) التي تمكن الاستدلال السريع بخطوات قليلة على أي LDMs مسبقة التدريب، بما في ذلك Stable Diffusion (rombach et al). من خلال النظر في عملية الانتشار العكسي الموجهة على أنها حل لمعادلة تفاضلية لاحتمالية التدفق المعززة (PF-ODE)، تم تصميم LCMs للتنبؤ المباشر بحل هذه المعادلة التفاضلية في الفضاء الكامن، مما يقلل الحاجة إلى العديد من التكرارات ويسمح بالمعاينة السريعة وعالية الدقة. يمكن تدريب LCM عالي الجودة بحجم 768 × 768 2-4 خطوة، والمقطرة بكفاءة من نماذج الانتشار الموجهة الخالية من المصنفات مسبقة التدريب، في غضون 32 ساعة فقط من معالج رسومات A100 GPU. علاوة على ذلك، نقدم طريقة جديدة تسمى Latent Consistency Fine-tuning (LCF) مصممة خصيصًا لضبط نماذج LCM على مجموعات بيانات الصور المخصصة. يظهر التقييم على مجموعة بيانات LAION-5B-Aesthetics أن نماذج LCM تحقق أداءً رائدًا على مستوى الدولة في توليد الصور النصية مع الاستدلال متعدد الخطوات."

يمكن العثور على عرض توضيحي لـ [SimianLuo/LCM_Dreamshaper_v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) checkpoint [هنا](https://huggingface.co/spaces/SimianLuo/Latent_Consistency_Model).

تمت المساهمة في الأنابيب بواسطة [luosiallen](https://luosiallen.github.io/) و [nagolinc](https://github.com/nagolinc) و [dg845](https://github.com/dg845).

## LatentConsistencyModelPipeline

[[autodoc]] LatentConsistencyModelPipeline

- all
- __call__
- enable_freeu
- disable_freeu
- enable_vae_slicing
- disable_vae_slicing
- enable_vae_tiling
- disable_vae_tiling

## LatentConsistencyModelImg2ImgPipeline

[[autodoc]] LatentConsistencyModelImg2ImgPipeline

- all
- __call__
- enable_freeu
- disable_freeu
- enable_vae_slicing
- disable_vae_slicing
- enable_vae_tiling
- disable_vae_tiling

## StableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput