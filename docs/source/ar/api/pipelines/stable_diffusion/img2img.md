# Image-to-image

يمكن أيضًا تطبيق نموذج Stable Diffusion على التوليد من صورة إلى صورة من خلال تمرير موجه نص وصورة أولية لشروط توليد الصور الجديدة.

يستخدم [`StableDiffusionImg2ImgPipeline`] آلية إزالة التشويش بالانتشار المقترحة في [SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://huggingface.co/papers/2108.01073) بواسطة Chenlin Meng، Yutong He، Yang Song، Jiaming Song، Jiajun Wu، Jun-Yan Zhu، Stefano Ermon.

المستخلص من الورقة هو:

*تمكّن عملية التوليد الموجه للصور المستخدمين العاديين من إنشاء وتحرير صور واقعية المظهر بأقل جهد ممكن. يتمثل التحدي الرئيسي في تحقيق التوازن بين الإخلاص لإدخال المستخدم (مثل السكتات الدماغية الملونة المرسومة باليد) وواقعية الصورة المولدة. تحاول الأساليب القائمة على GAN تحقيق هذا التوازن باستخدام GANs الشرطية أو عكس GANs، والتي تكون صعبة وغالبا ما تتطلب بيانات تدريب إضافية أو دالات خسارة لتطبيقات فردية. ولمعالجة هذه القضايا، نقدم طريقة جديدة لتوليف الصور وتحريرها، تسمى Stochastic Differential Editing (SDEdit)، بناءً على نموذج انتشار مولد مسبقًا، والذي يقوم بتوليف صور واقعية عن طريق إزالة التشويش بشكل تكراري من خلال معادلة تفاضلية عشوائية (SDE). نظرًا لصورة الإدخال مع دليل المستخدم من أي نوع، يقوم SDEdit أولاً بإضافة ضوضاء إلى الإدخال، ثم يقوم بإزالة تشويش الصورة الناتجة من خلال SDE المسبق لزيادة واقعيتها. لا يتطلب SDEdit تدريبًا أو عكسًا محددًا للمهمة ويمكنه تحقيق التوازن بين الواقعية والإخلاص بشكل طبيعي. يتفوق SDEdit بشكل كبير على أساليب GAN المستندة إلى حالة الفن بنسبة تصل إلى 98.09٪ في الواقعية و 91.72٪ في درجات الرضا الإجمالية، وفقًا لدراسة إدراكية بشرية، في مهام متعددة، بما في ذلك توليف الصور وتحريرها القائم على السكتة الدماغية بالإضافة إلى تركيب الصور.*

<Tip>
تأكد من الاطلاع على قسم Stable Diffusion [Tips](overview#tips) لمعرفة كيفية استكشاف المقايضة بين سرعة الجدولة والجودة، وكيفية إعادة استخدام مكونات الأنابيب بكفاءة!
</Tip>

## StableDiffusionImg2ImgPipeline

[[autodoc]] StableDiffusionImg2ImgPipeline
- all
- __call__
- enable_attention_slicing
- disable_attention_slicing
- enable_xformers_memory_efficient_attention
- disable_xformers_memory_efficient_attention
- load_textual_inversion
- from_single_file
- load_lora_weights
- save_lora_weights

## StableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput

## FlaxStableDiffusionImg2ImgPipeline

[[autodoc]] FlaxStableDiffusionImg2ImgPipeline
- all
- __call__

## FlaxStableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput