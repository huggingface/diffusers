# Consistency Models نماذج الاتساق
تم اقتراح نماذج الاتساق [Consistency Models](https://huggingface.co/papers/2303.01469)  في نماذج الاتساق من قبل يانغ سونغ وبرافولا داريوال ومارك تشين وإيليا سوتسكيفر.

الملخص من الورقة هو:

*لقد طورت نماذج الانتشار بشكل كبير مجالات توليد الصور والصوت والفيديو ، لكنها تعتمد على عملية أخذ العينات التكرارية التي تسبب التوليد البطيء. للتغلب على هذا القيد ، نقترح نماذج الاتساق ، وهي عائلة جديدة من النماذج التي تولد عينات عالية الجودة عن طريق تعيين الضوضاء مباشرة للبيانات. إنها تدعم التوليد السريع بخطوة واحدة حسب التصميم ، مع السماح بأخذ العينات متعددة الخطوات لتداول الحوسبة للحصول على جودة العينة. كما أنها تدعم تحرير البيانات بدون لقطة ، مثل طلاء الصور والتلوين والدقة الفائقة ، دون الحاجة إلى تدريب صريح على هذه المهام. يمكن تدريب نماذج الاتساق إما عن طريق تقطير نماذج الانتشار المدربة مسبقا ، أو كنماذج توليدية قائمة بذاتها تماما. من خلال التجارب المكثفة ، أثبتنا أنها تتفوق على تقنيات التقطير الحالية لنماذج الانتشار في أخذ العينات من خطوة واحدة وبضع خطوات ، وتحقيق FID الجديد المتطور البالغ 3.55 على CIFAR-10 و 6.20 على ImageNet 64x64 للجيل من خطوة واحدة. عندما يتم تدريبها بمعزل عن بعضها البعض ، تصبح نماذج الاتساق عائلة جديدة من النماذج التوليدية التي يمكن أن تتفوق على النماذج التوليدية الحالية ذات الخطوة الواحدة وغير العدائية على المعايير القياسية مثل CIFAR-10 و ImageNet 64x64 و LSUN 256x256.*

يمكن العثور على قاعدة الشفرة الأصلية في [openai/consistency_models](https://github.com/openai/consistency_models) ، وتتوفر نقاط تفتيش إضافية في openai.

The pipeline was contributed by [dg845](https://github.com/dg845) and [ayushtues](https://huggingface.co/ayushtues). ❤️


## Tips

لمزيد من التسريع، استخدم `torch.compile` لتوليد صور متعددة في <1 ثانية:

```diff
import torch
from diffusers import ConsistencyModelPipeline

device = "cuda"
# تحميل نقطة تفتيش cd_bedroom256_lpips.
model_id_or_path = "openai/diffusers-cd_bedroom256_lpips"
pipe = ConsistencyModelPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.to(device)

+ pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

# المعاينة متعددة الخطوات
# يمكن تحديد خطوات الوقت بشكل صريح؛ خطوات الوقت المحددة أدناه مأخوذة من مستودع GitHub الأصلي:
# https://github.com/openai/consistency_models/blob/main/scripts/launch.sh#L83
for _ in range(10):
    image = pipe(timesteps=[17, 0]).images[0]
    image.show()
```

## ConsistencyModelPipeline

[[autodoc]] ConsistencyModelPipeline

- all
- __call__

## ImagePipelineOutput

[[autodoc]] pipelines.ImagePipelineOutput