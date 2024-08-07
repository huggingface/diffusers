## تقييم نماذج الانتشار 

تقييم النماذج التوليدية مثل [Stable Diffusion](https://huggingface.co/docs/diffusers/stable_diffusion) هو أمر ذاتي بطبيعته. ولكن كممارسين وباحثين، غالباً ما يتعين علينا اتخاذ خيارات دقيقة من بين العديد من الاحتمالات المختلفة. لذلك، عند العمل مع نماذج توليدية مختلفة (مثل GANs و Diffusion، إلخ)، كيف نختار أحدها على الآخر؟

يمكن أن يكون التقييم النوعي لمثل هذه النماذج عرضة للأخطاء وقد يؤثر بشكل غير صحيح على القرار. ومع ذلك، فإن المقاييس الكمية لا تتوافق بالضرورة مع جودة الصورة. لذلك، عادة ما يوفر مزيج من التقييمات النوعية والكمية إشارة أقوى عند اختيار نموذج واحد على الآخر.

في هذه الوثيقة، نقدم نظرة عامة غير شاملة على الأساليب النوعية والكمية لتقييم نماذج الانتشار. وبالنسبة للأساليب الكمية، نركز بشكل خاص على كيفية تنفيذها إلى جانب `diffusers`.

يمكن أيضًا استخدام الأساليب الموضحة في هذه الوثيقة لتقييم مخططات الضوضاء المختلفة مع تثبيت نموذج التوليد الأساسي.

## السيناريوهات

نغطي نماذج الانتشار مع خطوط الأنابيب التالية:

- توليد الصور الموجهة بالنص (مثل [`StableDiffusionPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/text2img)).
- توليد الصور الموجهة بالنص، المشروطة أيضًا بصورة المدخلات (مثل [`StableDiffusionImg2ImgPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/img2img) و [`StableDiffusionInstructPix2PixPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/pix2pix)).
- نماذج توليد الصور المشروطة بالفصل (مثل [`DiTPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/dit)).

## التقييم النوعي

ينطوي التقييم النوعي عادة على تقييم بشري للصور المولدة. وتُقاس الجودة عبر جوانب مثل التكوين، والمحاذاة بين الصورة والنص، والعلاقات المكانية. توفر المطالبات الشائعة درجة من التوحيد للمقاييس الذاتية.

DrawBench و PartiPrompts هي مجموعات بيانات المطالبة المستخدمة للمقارنة المرجعية النوعية. تم تقديم DrawBench و PartiPrompts بواسطة [Imagen](https://imagen.research.google/) و [Parti](https://parti.research.google/) على التوالي.

من [الموقع الرسمي لـ Parti](https://parti.research.google/):

> PartiPrompts (P2) هي مجموعة غنية تضم أكثر من 1600 مطالبة باللغة الإنجليزية نقوم بإصدارها كجزء من هذا العمل. يمكن استخدام P2 لقياس قدرات النموذج عبر مختلف الفئات وتحديات الجوانب.

![parti-prompts](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/evaluation_diffusion_models/parti-prompts.png)

يحتوي PartiPrompts على الأعمدة التالية:

- مطالبة
- فئة المطالبة (مثل "مجردة"، "معرفة العالم"، إلخ)
- التحدي الذي يعكس الصعوبة (مثل "أساسي"، "معقد"، "الكتابة والرموز"، إلخ)

تسمح هذه المعايير المرجعية بالتقييم البشري جنبًا إلى جنب لنماذج توليد الصور المختلفة.

لهذا، قام فريق 🧨 Diffusers ببناء **Open Parti Prompts**، وهو معيار مرجعي نوعي مدفوع بالمجتمع يعتمد على Parti Prompts لمقارنة أحدث نماذج الانتشار مفتوحة المصدر:

- [Open Parti Prompts Game](https://huggingface.co/spaces/OpenGenAI/open-parti-prompts): لعشرة مطالبات من Parti، يتم عرض أربع صور، ويختار المستخدم الصورة التي تناسب المطالبة بشكل أفضل.
- [Open Parti Prompts Leaderboard](https://huggingface.co/spaces/OpenGenAI/parti-prompts-leaderboard): لوحة القيادة التي تقارن أفضل نماذج الانتشار مفتوحة المصدر في الوقت الحالي ببعضها البعض.

لمقارنة الصور يدويًا، دعنا نرى كيف يمكننا استخدام `diffusers` على بعض PartiPrompts.

فيما يلي بعض المطالبات التي تم أخذ عينات منها عبر تحديات مختلفة: Basic و Complex و Linguistic Structures و Imagination و Writing & Symbols. هنا نستخدم PartiPrompts كـ [مجموعة بيانات](https://huggingface.co/datasets/nateraw/parti-prompts).

```python
from datasets import load_dataset

# prompts = load_dataset("nateraw/parti-prompts", split="train")
# prompts = prompts.shuffle()
# sample_prompts = [prompts[i]["Prompt"] for i in range(5)]

# تثبيت هذه المطالبات التوضيحية لصالح القابلية للتكرار.
sample_prompts = [
"a corgi",
"a hot air balloon with a yin-yang symbol, with the moon visible in the daytime sky",
"a car with no windows",
"a cube made of porcupine",
'The saying "BE EXCELLENT TO EACH OTHER" written on a red brick wall with a graffiti image of a green alien wearing a tuxedo. A yellow fire hydrant is on a sidewalk in the foreground.',
]
```

الآن يمكننا استخدام هذه المطالبات لتوليد بعض الصور باستخدام Stable Diffusion ([v1-4 checkpoint](https://huggingface.co/CompVis/stable-diffusion-v1-4)):

```python
import torch

seed = 0
generator = torch.manual_seed(seed)

images = sd_pipeline(sample_prompts, num_images_per_prompt=1, generator=generator).images
```

![parti-prompts-14](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/evaluation_diffusion_models/parti-prompts-14.png)

يمكننا أيضًا تعيين `num_images_per_prompt` وفقًا لذلك لمقارنة صور مختلفة لنفس المطالبة. يؤدي تشغيل نفس خط الأنابيب ولكن مع نقطة مرجعية مختلفة ([v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)) إلى ما يلي:

![parti-prompts-15](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/evaluation_diffusion_models/parti-prompts-15.png)

بمجرد توليد عدة صور من جميع المطالبات باستخدام نماذج متعددة (قيد التقييم)، يتم تقديم هذه النتائج إلى مقيمين بشريين للتصنيف. لمزيد من التفاصيل حول معايير DrawBench و PartiPrompts المرجعية، يرجى الرجوع إلى أوراقهم البحثية.

<Tip>
من المفيد النظر في بعض عينات الاستدلال أثناء تدريب النموذج لقياس التقدم المحرز في التدريب. في [سكريبتات التدريب](https://github.com/huggingface/diffusers/tree/main/examples/) الخاصة بنا، ندعم هذه الأداة الوظيفية مع دعم إضافي للتسجيل في TensorBoard و Weights & Biases.
</Tip>

## التقييم الكمي

في هذا القسم، سنقوم بإرشادك خلال عملية تقييم ثلاث خطوط أنابيب انتشار مختلفة باستخدام:

- نتيجة CLIP
- التشابه الاتجاهي CLIP
- FID

### توليد الصور الموجهة بالنص

[نتيجة CLIP](https://arxiv.org/abs/2104.08718) تقيس توافق أزواج الصور والتعليقات التوضيحية. تشير الدرجات الأعلى لنتيجة CLIP إلى توافق أعلى 🔼. نتيجة CLIP هي قياس كمي للمفهوم النوعي "التوافق". يمكن أيضًا اعتبار توافق أزواج الصور والتعليقات التوضيحية على أنه التشابه الدلالي بين الصورة والتعليق التوضيحي. وقد وجد أن نتيجة CLIP لها علاقة قوية بالحكم البشري.

قم أولاً بتحميل [`StableDiffusionPipeline`]:

```python
from diffusers import StableDiffusionPipeline
import torch

model_ckpt = "CompVis/stable-diffusion-v1-4"
sd_pipeline = StableDiffusionPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16).to("cuda")
```

قم بتوليد بعض الصور مع مطالبات متعددة:

```python
prompts = [
"a photo of an astronaut riding a horse on mars",
"A high tech solarpunk utopia in the Amazon rainforest",
"A pikachu fine dining with a view to the Eiffel Tower",
"A mecha robot in a favela in expressionist style",
"an insect robot preparing a delicious meal",
"A small cabin on top of a snowy mountain in the style of Disney, artstation",
]

images = sd_pipeline(prompts, num_images_per_prompt=1, output_type="np").images

print(images.shape)
# (6, 512, 512, 3)
```

بعد ذلك، نقوم بحساب نتيجة CLIP.

```python
from torchmetrics.functional.multimodal import clip_score
from functools import partial

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts):
images_int = (images * 255).astype("uint8")
clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
return round(float(clip_score), 4)

sd_clip_score = calculate_clip_score(images, prompts)
print(f"CLIP score: {sd_clip_score}")
# نتيجة CLIP: 35.7038
```

في المثال أعلاه، قمنا بتوليد صورة واحدة لكل مطالبة. إذا قمنا بتوليد صور متعددة لكل مطالبة، فسوف نضطر إلى أخذ متوسط الدرجات من الصور المولدة لكل مطالبة.

الآن، إذا أردنا مقارنة نقطتي مراقبة متوافقتين مع [`StableDiffusionPipeline`]، فيجب علينا تمرير مولد أثناء استدعاء خط الأنابيب. أولاً، نقوم بتوليد الصور باستخدام بذرة ثابتة مع [نقطة مرجعية مستقرة للانتشار v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4):

```python
seed = 0
generator = torch.manual_seed(seed)

images = sd_pipeline(prompts, num_images_per_prompt=1, generator=generator, output_type="np").images
```

بعد ذلك، نقوم بتحميل نقطة مرجعية [v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) لتوليد الصور:

```python
model_ckpt_1_5 = "runwayml/stable-diffusion-v1-5"
sd_pipeline_1_5 = StableDiffusionPipeline.from_pretrained(model_ckpt_1_5, torch_dtype=weight_dtype).to(device)

images_1_5 = sd_pipeline_1_5(prompts, num_images_per_prompt=1, generator=generator, output_type="np").images
```

وأخيرًا، نقارن نتائج CLIP الخاصة بهم:

```python
sd_clip_score_1_4 = calculate_clip_score(images, prompts)
print(f"نتيجة CLIP مع v-1-4: {sd_clip_score_1_4}")
# نتيجة CLIP مع v-1-4: 34.9102

sd_clip_score_1_5 = calculate_clip_score(images_1_5, prompts)
print(f"نتيجة CLIP مع v-1-5: {sd_clip_score_1_5}")
# نتيجة CLIP مع v-1-5: 36.2137
```

يبدو أن نقطة مرجعية [v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) تؤدي أداءً أفضل من سابقتها. ومع ذلك، لاحظ أن عدد المطالبات التي استخدمناها لحساب نتائج CLIP منخفضة جدًا. من أجل تقييم أكثر عملية، يجب أن يكون هذا الرقم أعلى بكثير، ويجب أن تكون المطالبات متنوعة.

<Tip warning={true}>
بحكم البناء، هناك بعض القيود في هذه النتيجة. كانت التعليقات التوضيحية في مجموعة البيانات التدريبية
تم الحصول عليها من الويب وتم استخراجها من العلامات `alt` والعلامات المماثلة المرتبطة بصورة على الإنترنت.
قد لا تكون ممثلة بالضرورة لما قد يستخدمه الإنسان لوصف صورة. وبالتالي، كان علينا "هندسة" بعض المطالبات هنا.
</Tip>
### توليد الصور المشروطة بالصور والنصوص

في هذه الحالة، نقوم بضبط خط أنابيب التوليد باستخدام صورة إدخال بالإضافة إلى موجه نصي. دعنا نأخذ [`StableDiffusionInstructPix2PixPipeline`] كمثال. فهو يأخذ تعليمات تعديل كإدخال موجه وصورة إدخال ليتم تعديلها.

هذا مثال على ذلك:

![تعليمات التعديل](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/evaluation_diffusion_models/edit-instruction.png)

تتمثل إحدى الاستراتيجيات لتقييم مثل هذا النموذج في قياس اتساق التغيير بين الصورتين (في مساحة CLIP) مع التغيير بين تعليقي الصورتين (كما هو موضح في "تكييف المجال الموجه بـ CLIP لمولدات الصور"). يشار إلى ذلك باسم "**تشابه الاتجاه CLIP**".

- يتوافق العنوان الفرعي 1 مع صورة الإدخال (الصورة 1) التي سيتم تعديلها.
- يتوافق العنوان الفرعي 2 مع الصورة المعدلة (الصورة 2). يجب أن يعكس تعليمات التعديل.

فيما يلي نظرة عامة توضيحية:

![اتساق التعديل](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/evaluation_diffusion_models/edit-consistency.png)

لقد أعددنا مجموعة بيانات مصغرة لتنفيذ هذا المقياس. دعنا أولاً نقوم بتحميل مجموعة البيانات.

```python
from datasets import load_dataset

dataset = load_dataset("sayakpaul/instructpix2pix-demo", split="train")
dataset.features
```

```bash
{'input': Value(dtype='string', id=None),
'edit': Value(dtype='string', id=None),
'output': Value(dtype='string', id=None),
'image': Image(decode=True, id=None)}
```

هنا لدينا:

- `input` هو عنوان فرعي مطابق لـ `image`.
- يشير `edit` إلى تعليمات التعديل.
- يشير `output` إلى العنوان الفرعي المعدل الذي يعكس تعليمات `edit`.

دعنا نلقي نظرة على عينة.

```python
idx = 0
print(f"Original caption: {dataset[idx]['input']}")
print(f"Edit instruction: {dataset[idx]['edit']}")
print(f"Modified caption: {dataset[idx]['output']}")
```

```bash
Original caption: 2. FAROE ISLANDS: An archipelago of 18 mountainous isles in the North Atlantic Ocean between Norway and Iceland, the Faroe Islands has 'everything you could hope for', according to Big 7 Travel. It boasts 'crystal clear waterfalls, rocky cliffs that seem to jut out of nowhere and velvety green hills'
Edit instruction: make the isles all white marble
Modified caption: 2. WHITE MARBLE ISLANDS: An archipelago of 18 mountainous white marble isles in the North Atlantic Ocean between Norway and Iceland, the White Marble Islands has 'everything you could hope for', according to Big 7 Travel. It boasts 'crystal clear waterfalls, rocky cliffs that seem to jut out of nowhere and velvety green hills'
```

وهذه هي الصورة:

```python
dataset[idx]["image"]
```

![مجموعة بيانات التعديل](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/evaluation_diffusion_models/edit-dataset.png)

سنقوم أولاً بتعديل صور مجموعة البيانات الخاصة بنا باستخدام تعليمات التعديل وحساب التشابه الاتجاهي.

دعنا أولاً نقوم بتحميل [`StableDiffusionInstructPix2PixPipeline`]:

```python
from diffusers import StableDiffusionInstructPix2PixPipeline

instruct_pix2pix_pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
"timbrooks/instruct-pix2pix", torch_dtype=torch.float16
).to(device)
```

الآن، نقوم بالتعديلات:

```python
import numpy as np


def edit_image(input_image, instruction):
    image = instruct_pix2pix_pipeline(
    instruction,
    image=input_image,
    output_type="np",
    generator=generator,
    ).images[0]
    return image

input_images = []
original_captions = []
modified_captions = []
edited_images = []

for idx in range(len(dataset)):
    input_image = dataset[idx]["image"]
    edit_instruction = dataset[idx]["edit"]
    edited_image = edit_image(input_image, edit_instruction)

    input_images.append(np.array(input_image))
    original_captions.append(dataset[idx]["input"])
    modified_captions.append(dataset[idx]["output"])
    edited_images.append(edited_image)
```

لقياس التشابه الاتجاهي، نقوم أولاً بتحميل برامج الترميز للصور والنصوص CLIP:

```python
from transformers import (
CLIPTokenizer,
CLIPTextModelWithProjection,
CLIPVisionModelWithProjection,
CLIPImageProcessor,
)

clip_id = "openai/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(clip_id)
text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(device)
image_processor = CLIPImageProcessor.from_pretrained(clip_id)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(device)
```

لاحظ أننا نستخدم نقطة تفتيش CLIP معينة، أي `openai/clip-vit-large-patch14`. ويرجع ذلك إلى أن التدريب المسبق لـ Stable Diffusion تم إجراؤه باستخدام هذا المتغير من CLIP. لمزيد من التفاصيل، يرجى الرجوع إلى [الوثائق](https://huggingface.co/docs/transformers/model_doc/clip).

بعد ذلك، نقوم بإعداد `nn.Module` من PyTorch لحساب التشابه الاتجاهي:

```python
import torch.nn as nn
import torch.nn.functional as F


class DirectionalSimilarity(nn.Module):
    def __init__(self, tokenizer, text_encoder, image_processor, image_encoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.image_encoder = image_encoder

    def preprocess_image(self, image):
        image = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": image.to(device)}

    def tokenize_text(self, text):
        inputs = self.tokenizer(
        text,
        max_length=self.tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.to(device)}

    def encode_image(self, image):
        preprocessed_image = self.preprocess_image(image)
        image_features = self.image_encoder(**preprocessed_image).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def encode_text(self, text):
        tokenized_text = self.tokenize_text(text)
        text_features = self.text_encoder(**tokenized_text).text_embeds
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def compute_directional_similarity(self, img_feat_one, img_feat_two, text_feat_one, text_feat_two):
        sim_direction = F.cosine_similarity(img_feat_two - img_feat_one, text_feat_two - text_feat_one)
        return sim_direction

    def forward(self, image_one, image_two, caption_one, caption_two):
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)
        text_feat_one = self.encode_text(caption_one)
        text_feat_two = self.encode_text(caption_two)
        directional_similarity = self.compute_directional_similarity(
        img_feat_one, img_feat_two, text_feat_one, text_feat_two
        )
        return directional_similarity
```

دعنا نستخدم `DirectionalSimilarity` الآن.

```python
dir_similarity = DirectionalSimilarity(tokenizer, text_encoder, image_processor, image_encoder)
scores = []

for i in range(len(input_images)):
    original_image = input_images[i]
    original_caption = original_captions[i]
    edited_image = edited_images[i]
    modified_caption = modified_captions[i]

    similarity_score = dir_similarity(original_image, edited_image, original_caption, modified_caption)
    scores.append(float(similarity_score.detach().cpu()))

print(f"CLIP directional similarity: {np.mean(scores)}")
# تشابه الاتجاه CLIP: 0.0797976553440094
```

مثل درجة CLIP، كلما كانت درجة التشابه الاتجاهي CLIP أعلى، كان ذلك أفضل.

تجدر الإشارة إلى أن `StableDiffusionInstructPix2PixPipeline` تعرض حجتين، وهما `image_guidance_scale` و`guidance_scale`، واللتان تتيحان لك التحكم في جودة الصورة النهائية المعدلة. نشجعك على تجربة هاتين الحجتين ورؤية تأثيرهما على التشابه الاتجاهي.

يمكننا توسيع فكرة هذا المقياس لقياس مدى تشابه الصورة الأصلية والنسخة المعدلة. للقيام بذلك، يمكننا ببساطة إجراء `F.cosine_similarity (img_feat_two، img_feat_one)`. بالنسبة لهذه التعديلات، ما زلنا نريد الحفاظ على الدلالات الأساسية للصور قدر الإمكان، أي درجة تشابه عالية.

يمكننا استخدام هذه المقاييس لخطوط الأنابيب المماثلة مثل [`StableDiffusionPix2PixZeroPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/pix2pix_zero#diffusers.StableDiffusionPix2PixZeroPipeline).

<Tip>
يعتمد كل من درجة CLIP والتشابه الاتجاهي CLIP على نموذج CLIP، مما قد يجعل التقييمات متحيزة.
</Tip>

***يمكن أن يكون تمديد المقاييس مثل IS، FID (التي تمت مناقشتها لاحقًا)، أو KID أمرًا صعبًا*** عندما يكون النموذج قيد التقييم قد تم تدريبه مسبقًا على مجموعة بيانات كبيرة للصور والتعليقات التوضيحية (مثل [مجموعة بيانات LAION-5B](https://laion.ai/blog/laion-5b/)). ويرجع ذلك إلى أن هذه المقاييس تعتمد على شبكة InceptionNet (المدربة مسبقًا على مجموعة بيانات ImageNet-1k) المستخدمة لاستخراج ميزات الصور المتوسطة. قد يكون هناك تداخل محدود بين مجموعة بيانات التدريب المسبق لـ Stable Diffusion ومجموعة بيانات التدريب المسبق لـ InceptionNet، لذلك فإنها ليست مرشحًا جيدًا هنا لاستخراج الميزات.

***يساعد استخدام المقاييس المذكورة أعلاه في تقييم النماذج المشروطة بالفصل. على سبيل المثال، [DiT](https://huggingface.co/docs/diffusers/main/en/api/pipelines/dit). تم تدريبه مسبقًا بشرط أن يكون مشروطًا بفصول ImageNet-1k.***
### النماذج التوليدية المشروطة بالفئة

عادةً ما يتم التدريب المسبق للنماذج التوليدية المشروطة بالفئة على مجموعة بيانات موسومة بالفئات مثل [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k). تشمل المقاييس الشائعة لتقييم هذه النماذج مسافة فرشيه انسياب الفرق (FID)، ومسافة انسياب النواة (KID)، ودرجة الانسياب (IS). وفي هذه الوثيقة، نركز على FID ([Heusel et al.](https://arxiv.org/abs/1706.08500)). وسنوضح كيفية حسابها باستخدام [`DiTPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/dit)، والتي تستخدم [نموذج DiT](https://arxiv.orgMultiplier-free Transformer for Image Generation) تحت الغطاء.

تهدف FID إلى قياس مدى تشابه مجموعتين من الصور. وفقًا [لهذا المورد](https://mmgeneration.readthedocs.io/en/latest/quick_run.html#fid):

> تعد مسافة فرشيه انسياب الفرق مقياسًا لتشابه مجموعتين من الصور. وقد ثبت أنها تتوافق جيدًا مع الحكم البشري على الجودة المرئية، ويتم استخدامها في أغلب الأحيان لتقييم جودة العينات التي تنتجها الشبكات الخصمية التوليدية. يتم حساب FID عن طريق حساب مسافة فرشيه بين اثنين من التوزيعات الاحتمالية الغاوسية المناسبة لتمثيل الخصائص التي ينتجها نموذج Inception.

تتمثل هاتان المجموعتان من البيانات بشكل أساسي في مجموعة بيانات الصور الحقيقية ومجموعة بيانات الصور المزيفة (الصور المولدة في حالتنا). وعادة ما يتم حساب FID باستخدام مجموعتين كبيرتين من البيانات. ومع ذلك، سنعمل في هذه الوثيقة مع مجموعتين صغيرتين من البيانات.

دعونا نقوم أولاً بتنزيل بعض الصور من مجموعة بيانات التدريب ImageNet-1k:

```python
from zipfile import ZipFile
import requests


def download(url, local_filepath):
    r = requests.get(url)
    with open(local_filepath, "wb") as f:
        f.write(r.content)
    return local_filepath

dummy_dataset_url = "https://hf.co/datasets/sayakpaul/sample-datasets/resolve/main/sample-imagenet-images.zip"
local_filepath = download(dummy_dataset_url, dummy_dataset_url.split("/")[-1])

with ZipFile(local_filepath, "r") as zipper:
    zipper.extractall(".")
```

```python
from PIL import Image
import os

dataset_path = "sample-imagenet-images"
image_paths = sorted([os.path.join(dataset_path, x) for x in os.listdir(dataset_path)])

real_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]
```

هذه 10 صور من فئات ImageNet-1k التالية: "cassette_player"، "chain_saw" (x2)، "church"، "gas_pump" (x3)، "parachute" (x2)، و "tench".

<p align="center">
<img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/evaluation_diffusion_models/real-images.png" alt="real-images"><br>
<em>Real images.</em>
</p>

الآن بعد تحميل الصور، دعنا نطبق بعض المعالجة المسبقة البسيطة عليها لاستخدامها في حساب FID.

```python
from torchvision.transforms import functional as F


def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return F.center_crop(image, (256, 256))

real_images = torch.cat([preprocess_image(image) for image in real_images])
print(real_images.shape)
# torch.Size([10, 3, 256, 256])
```

سنقوم الآن بتحميل [`DiTPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/dit) لتوليد الصور المشروطة بالفئات المذكورة أعلاه.

```python
from diffusers import DiTPipeline, DPMSolverMultistepScheduler

dit_pipeline = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
dit_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(dit_pipeline.scheduler.config)
dit_pipeline = dit_pipeline.to("cuda")

words = [
    "cassette player",
    "chainsaw",
    "chainsaw",
    "church",
    "gas pump",
    "gas pump",
    "gas pump",
    "parachute",
    "parachute",
    "tench",
]

class_ids = dit_pipeline.get_label_ids(words)
output = dit_pipeline(class_labels=class_ids, generator=generator, output_type="np")

fake_images = output.images
fake_images = torch.tensor(fake_images)
fake_images = fake_images.permute(0, 3, 1, 2)
print(fake_images.shape)
# torch.Size([10, 3, 256, 256])
```

الآن، يمكننا حساب FID باستخدام [`torchmetrics`](https://torchmetrics.readthedocs.io/).

```python
from torchmetrics.image.fid import FrechetInceptionDistance

fid = FrechetInceptionDistance(normalize=True)
fid.update(real_images, real=True)
fid.update(fake_images, real=False)

print(f"FID: {float(fid.compute())}")
# FID: 177.7147216796875
```

كلما انخفضت قيمة FID، كان ذلك أفضل. يمكن أن يؤثر العديد من العوامل على FID هنا:

- عدد الصور (الحقيقية والمزيفة)
- العشوائية في عملية الانتشار
- عدد خطوات الاستدلال في عملية الانتشار
- الجدولة المستخدمة في عملية الانتشار

بالنسبة للنقطتين الأخيرتين، من الجيد إذن إجراء التقييم عبر بذور وخطوات استدلال مختلفة، ثم الإبلاغ عن نتيجة متوسطة.

<Tip warning={true}>

تميل نتائج FID إلى أن تكون هشة لأنها تعتمد على العديد من العوامل:

* نموذج Inception المحدد المستخدم أثناء الحساب.
* دقة تنفيذ الحساب.
* تنسيق الصورة (ليس نفسه إذا بدأنا من PNGs مقابل JPGs).

مع مراعاة ذلك، تكون FID مفيدة في كثير من الأحيان عند مقارنة الجولات المماثلة، ولكن من الصعب استنساخ نتائج الورقة ما لم يكشف المؤلفون بعناية عن رمز قياس FID.

تنطبق هذه النقاط أيضًا على المقاييس الأخرى ذات الصلة، مثل KID و IS.

</Tip>

كخطوة أخيرة، دعنا نقوم بالتفتيش البصري على `fake_images`.

<p align="center">
<img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/evaluation_diffusion_models/fake-images.png" alt="fake-images"><br>
<em>Fake images.</em>
</p>