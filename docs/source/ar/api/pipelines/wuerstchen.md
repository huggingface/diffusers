# Würstchen

![صورة](https://github.com/dome272/Wuerstchen/assets/61938694/0617c863-165a-43ee-9303-2a17299a0cf9)

[Wuerstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models](https://huggingface.co/papers/2306.00637) هو من تأليف بابلو بيرنياس، ودومينيك رامباس، وماتس إل. ريختر، وكريستوفر بال، ومارك أوبرفيل.

الملخص المستخرج من الورقة هو:

*نحن نقدم Würstchen، تصميم جديد لتركيب الصور النصية يجمع بين الأداء التنافسي وفعالية التكلفة غير المسبوقة لنماذج الانتشار النصي للصور واسعة النطاق. تتمثل المساهمة الرئيسية لعملنا في تطوير تقنية انتشار الكامن حيث نتعلم تمثيلًا دقيقًا ولكنه مضغوط للغاية للصورة الدلالية المستخدمة لتوجيه عملية الانتشار. يوفر هذا التمثيل عالي الضغط لصورة إرشادات أكثر تفصيلاً مقارنة بالتمثيلات الكامنة للغة، مما يقلل بشكل كبير من المتطلبات الحسابية لتحقيق نتائج متقدمة. كما يحسن نهجنا جودة توليد الصور المشروطة بالنص بناءً على دراستنا التفضيلية للمستخدم. وتتكون متطلبات التدريب لنهجنا من 24602 ساعة من معالج A100 GPU - مقارنة بـ 200000 ساعة من معالج GPU لبرنامج Stable Diffusion 2.1. كما يتطلب نهجنا كمية أقل من بيانات التدريب لتحقيق هذه النتائج. علاوة على ذلك، تسمح لنا التمثيلات الكامنة المضغوطة بإجراء الاستدلال بسرعة أكبر مرتين، مما يقلل بشكل كبير من التكاليف والبصمة الكربونية المعتادة لنموذج الانتشار المتقدم، دون المساس بالأداء النهائي. في مقارنة أوسع مع النماذج المتقدمة، يكون نهجنا أكثر كفاءة إلى حد كبير ويحظى بتقدير إيجابي من حيث جودة الصورة. نعتقد أن هذا العمل يحفز على زيادة التركيز على إعطاء الأولوية لكل من الأداء وإمكانية الوصول الحسابي.*

## نظرة عامة على Würstchen

Würstchen هو نموذج انتشار، يعمل نموذج الشرطي النصي الخاص به في مساحة الكامن المضغوطة للغاية للصور. لماذا هذا مهم؟ يمكن أن يؤدي ضغط البيانات إلى تقليل التكاليف الحسابية لكل من التدريب والاستدلال بمقادير. التدريب على الصور 1024x1024 أكثر تكلفة بكثير من التدريب على 32x32. عادة، تستخدم الأعمال الأخرى ضغطًا صغيرًا نسبيًا، في نطاق 4x - 8x ضغط مكاني. Würstchen يأخذ هذا إلى أقصى حد. من خلال التصميم الجديد، نحقق ضغطًا مكانيًا يبلغ 42x. لم يشاهد هذا من قبل لأن الطرق الشائعة تفشل في إعادة بناء الصور التفصيلية بإخلاص بعد الضغط المكاني 16x. تستخدم Würstchen ضغطًا من مرحلتين، ما نسميه المرحلة أ والمرحلة ب. المرحلة أ هي VQGAN، والمرحلة ب هي Autoencoder Diffusion (يمكن العثور على مزيد من التفاصيل في [الورقة](https://huggingface.co/papers/2306.00637)). يتم تعلم نموذج ثالث، المرحلة ج، في تلك المساحة الكامنة المضغوطة للغاية. يتطلب هذا التدريب كسورًا من الحساب المستخدم للنماذج ذات الأداء الأفضل حاليًا، مع السماح أيضًا بالاستدلال الأرخص والأسرع.

## تأتي Würstchen v2 إلى أجهزة الانتشار

بعد الإصدار الأولي للورقة، قمنا بتحسين العديد من الأشياء في الهندسة المعمارية والتدريب وأخذ العينات، مما يجعل Würstchen تنافسية مع النماذج الحالية المتقدمة في العديد من الطرق. نحن متحمسون لإصدار هذه النسخة الجديدة مع أجهزة الانتشار. فيما يلي قائمة بالتحسينات.

- دقة أعلى (1024x1024 حتى 2048x2048)
- استدلال أسرع
- عينة متعددة الجوانب الدقة
- جودة أفضل

نقوم بإصدار 3 نقاط تفتيش لنموذج توليد الصور المشروط بالنص (المرحلة ج). وهي:

- v2-base
- v2-aesthetic
- **(default)** v2-interpolated (50% interpolation between v2-base and v2-aesthetic)

نوصي باستخدام v2-interpolated، حيث أنه يحتوي على لمسة لطيفة من الواقعية الفوتوغرافية والجمالية. استخدم v2-base للضبط الدقيق لأنه لا يحتوي على تحيز أسلوبي واستخدم v2-aesthetic للتوليدات الفنية للغاية.

يمكن رؤية المقارنة هنا:

![صورة](https://github.com/dome272/Wuerstchen/assets/61938694/2914830f-cbd3-461c-be64-d50734f4b49d "مقارنة بين إصدارات Würstchen")

## توليد النص إلى الصورة

من أجل سهولة الاستخدام، يمكن استخدام Würstchen مع خط أنابيب واحد. يمكن استخدام خط الأنابيب هذا على النحو التالي:

```python
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS

pipe = AutoPipelineForText2Image.from_pretrained("warp-ai/wuerstchen", torch_dtype=torch.float16).to("cuda")

caption = "Anthropomorphic cat dressed as a fire fighter"
images = pipe(
    caption,
    width=1024,
    height=1536,
    prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
    prior_guidance_scale=4.0,
    num_images_per_prompt=2,
).images
```

ولأغراض الشرح، يمكننا أيضًا تهيئة خطي الأنابيب الرئيسيين في Würstchen بشكل فردي. تتكون Würstchen من 3 مراحل: المرحلة ج، المرحلة ب، المرحلة أ. لديهم جميعًا وظائف مختلفة ولا تعمل إلا معًا. عندما يتم إنشاء الصور المشروطة بالنص، ستقوم المرحلة ج أولاً بتوليد الكامنات في مساحة الكامن المضغوطة للغاية. هذا ما يحدث في `prior_pipeline`. بعد ذلك، يتم تمرير الكامنات المولدة إلى المرحلة ب، والتي تقوم بفك ضغط الكامنات إلى مساحة كامنة أكبر لـ VQGAN. بعد ذلك، يمكن فك تشفير هذه الكامنات بواسطة المرحلة أ، والتي تعد VQGAN، إلى مساحة البكسل. يتم تضمين المرحلة ب والمرحلة أ في `decoder_pipeline`. لمزيد من التفاصيل، راجع [الورقة](https://huggingface.co/papers/2306.00637).

```python
import torch
from diffusers import WuerstchenDecoderPipeline, WuerstchenPriorPipeline
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS

device = "cuda"
dtype = torch.float16
num_images_per_prompt = 2

prior_pipeline = WuerstchenPriorPipeline.from_pretrained(
    "warp-ai/wuerstchen-prior", torch_dtype=dtype
).to(device)
decoder_pipeline = WuerstchenDecoderPipeline.from_pretrained(
    "warp-ai/wuerstchen", torch_dtype=dtype
).to(device)

caption = "Anthropomorphic cat dressed as a fire fighter"
negative_prompt = ""

prior_output = prior_pipeline(
    prompt=caption,
    height=1024,
    width=1536,
    timesteps=DEFAULT_STAGE_C_TIMESTEPS,
    negative_prompt=negative_prompt,
    guidance_scale=4.0,
    num_images_per_prompt=num_images_per_prompt,
)
decoder_output = decoder_pipeline(
    image_embeddings=prior_output.image_embeddings,
    prompt=caption,
    negative_prompt=negative_prompt,
    guidance_scale=0.0,
    output_type="pil",
).images[0]
decoder_output
```

## تسريع الاستدلال

يمكنك الاستفادة من وظيفة `torch.compile` والحصول على تسريع يبلغ حوالي 2-3x:

```python
prior_pipeline.prior = torch.compile(prior_pipeline.prior, mode="reduce-overhead", fullgraph=True)
decoder_pipeline.decoder = torch.compile(decoder_pipeline.decoder, mode="reduce-overhead", fullgraph=True)
```

## القيود

- بسبب الضغط العالي الذي تستخدمه Würstchen، يمكن أن تفتقر الأجيال إلى قدر جيد
من التفاصيل. بالنسبة لعين الإنسان، يكون هذا ملحوظًا بشكل خاص في الوجوه والأيدي وما إلى ذلك.
- **يمكن إنشاء الصور فقط في خطوات بكسل 128**، على سبيل المثال، الدقة الأعلى التالية
بعد 1024x1024 هو 1152x1152
- يفتقر النموذج إلى القدرة على عرض النص بشكل صحيح في الصور
- غالبًا ما لا يحقق النموذج الواقعية الفوتوغرافية
- يصعب على النموذج التعامل مع المطالبات التركيبية الصعبة

يمكن العثور على الكود الأصلي، بالإضافة إلى الأفكار التجريبية، في [dome272/Wuerstchen](https://github.com/dome272/Wuerstchen).

## WuerstchenCombinedPipeline

[[autodoc]] WuerstchenCombinedPipeline

- all
- __call__

## WuerstchenPriorPipeline

[[autodoc]] WuerstchenPriorPipeline

- all
- __call__

## WuerstchenPriorPipelineOutput

[[autodoc]] pipelines.wuerstchen.pipeline_wuerstchen_prior.WuerstchenPriorPipelineOutput

## WuerstchenDecoderPipeline

[[autodoc]] WuerstchenDecoderPipeline

- all
- __call__

## الاستشهاد

```bibtex
@misc{pernias2023wuerstchen,
title={Wuerstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models},
author={Pablo Pernias and Dominic Rampas and Mats L. Richter and Christopher J. Pal and Marc Aubreville},
year={2023},
eprint={2306.00637},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
```