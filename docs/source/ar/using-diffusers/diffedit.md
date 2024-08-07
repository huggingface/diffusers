# DiffEdit 

[[open-in-colab]]

لا تتطلب عمليات تحرير الصور عادة سوى توفير قناع لمنطقة التحرير. يقوم DiffEdit تلقائيًا بإنشاء القناع لك بناءً على استعلام نصي، مما يسهل عملية إنشاء قناع بدون استخدام برامج تحرير الصور. تعمل خوارزمية DiffEdit في ثلاث خطوات:

1. يقوم نموذج الانتشار بتنقية صورة ما بناءً على نص استعلام ونص مرجعي، مما يؤدي إلى تقديرات ضوضاء مختلفة لمناطق مختلفة من الصورة؛ ويُستخدم الفرق لاستنتاج قناع لتحديد أي جزء من الصورة يحتاج إلى تغيير لمطابقة نص الاستعلام.

2. يتم ترميز الصورة المدخلة إلى مساحة الكامنة باستخدام DDIM.

3. يتم فك تشفير الكامنات باستخدام نموذج الانتشار المشروط على نص الاستعلام، باستخدام القناع كدليل بحيث تظل البكسلات خارج القناع كما هي في الصورة المدخلة.

سيوضح لك هذا الدليل كيفية استخدام DiffEdit لتحرير الصور دون إنشاء قناع يدويًا.

قبل البدء، تأكد من تثبيت المكتبات التالية:

```py
# قم بإلغاء التعليق لتثبيت المكتبات الضرورية في Colab
#! pip install -q diffusers transformers accelerate
```

يتطلب [`StableDiffusionDiffEditPipeline`] قناع صورة ومجموعة من الكامنات المعكوسة جزئيًا. يتم إنشاء قناع الصورة من الدالة [`~StableDiffusionDiffEditPipeline.generate_mask`]`، ويتضمن معلمتين، `source_prompt` و`target_prompt`. تحدد هذه المعلمات ما سيتم تحريره في الصورة. على سبيل المثال، إذا كنت تريد تغيير وعاء من *الفواكه* إلى وعاء من *الكمثرى*، فستكون:

```py
source_prompt = "وعاء من الفواكه"
target_prompt = "وعاء من الكمثرى"
```

يتم إنشاء الكامنات المعكوسة جزئيًا من الدالة [`~StableDiffusionDiffEditPipeline.invert`]`، ومن الجيد عمومًا تضمين `prompt` أو *caption* لوصف الصورة للمساعدة في توجيه عملية أخذ العينات العكسية للكامن. غالبًا ما يكون التعليق هو `source_prompt` الخاص بك، ولكن يمكنك تجربة أوصاف نصية أخرى!

قم بتحميل الأنبوب، ومخطط المعايرة، ومخطط المعايرة العكسية، وتمكين بعض التحسينات لتقليل استخدام الذاكرة:

```py
import torch
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionDiffEditPipeline

pipeline = StableDiffusionDiffEditPipeline.from_pretrained(
"stabilityai/stable-diffusion-2-1",
torch_dtype=torch.float16,
safety_checker=None,
use_safetensors=True,
)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_model_cpu_offload()
pipeline.enable_vae_slicing()
```

قم بتحميل الصورة التي تريد تحريرها:

```py
from diffusers.utils import load_image, make_image_grid

img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"
raw_image = load_image(img_url).resize((768, 768))
raw_image
```

استخدم الدالة [`~StableDiffusionDiffEditPipeline.generate_mask`] لإنشاء قناع الصورة. ستحتاج إلى تمرير `source_prompt` و`target_prompt` لتحديد ما سيتم تحريره في الصورة:

```py
from PIL import Image

source_prompt = "وعاء من الفواكه"
target_prompt = "سلة من الكمثرى"
mask_image = pipeline.generate_mask(
image=raw_image,
source_prompt=source_prompt,
target_prompt=target_prompt,
)
Image.fromarray((mask_image.squeeze()*255).astype("uint8"), "L").resize((768, 768))
```

الآن، قم بإنشاء الكامنات العكسية ومرر لها تعليقًا يصف الصورة:

```py
inv_latents = pipeline.invert(prompt=source_prompt, image=raw_image).latents
```

أخيرًا، قم بتمرير قناع الصورة والكامنات المعكوسة إلى الأنبوب. يصبح `target_prompt` هو `prompt` الآن، ويتم استخدام `source_prompt` كـ `negative_prompt`:

```py
output_image = pipeline(
prompt=target_prompt,
mask_image=mask_image,
image_latents=inv_latents,
negative_prompt=source_prompt,
).images[0]
mask_image = Image.fromarray((mask_image.squeeze()*255).astype("uint8"), "L").resize((768, 768))
make_image_grid([raw_image, mask_image, output_image], rows=1, cols=3)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأصلية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://github.com/Xiang-cd/DiffEdit-stable-diffusion/blob/main/assets/target.png?raw=true"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة بعد التحرير</figcaption>
</div>
</div>

## إنشاء تضمين المصدر والهدف

يمكن إنشاء تضمينات المصدر والهدف تلقائيًا باستخدام نموذج [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) بدلاً من إنشائها يدويًا.

قم بتحميل نموذج Flan-T5 ومصنف الرموز من مكتبة 🤗 Transformers:

```py
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto", torch_dtype=torch.float16)
```

قدم بعض النصوص الأولية لطلب نموذج لإنشاء مطالبات المصدر والهدف.

```py
source_concept = "bowl"
target_concept = "basket"

source_text = f"Provide a caption for images containing a {source_concept}. "
"The captions should be in English and should be no longer than 150 characters."

target_text = f"Provide a caption for images containing a {target_concept}. "
"The captions should be in English and should be no longer than 150 characters."
```

بعد ذلك، قم بإنشاء دالة مساعدة لإنشاء المطالبات:

```py
@torch.no_grad()
def generate_prompts(input_prompt):
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(
input_ids، درجة الحرارة = 0.8، num_return_sequences=16، do_sample=True، max_new_tokens=128، top_k=10
)
return tokenizer.batch_decode(outputs, skip_special_tokens=True)

source_prompts = generate_prompts(source_text)
target_prompts = generate_prompts(target_text)
print(source_prompts)
print(target_prompts)
```

<Tip>
إطلع على [إستراتيجيات التوليد](https://huggingface.co/docs/transformers/main/en/generation_strategies)
 إذا كنت مهتمًا بمعرفة المزيد عن استراتيجيات إنشاء نص ذي جودة مختلفة.
</Tip>

قم بتحميل نموذج الترميز النصي المستخدم بواسطة [`StableDiffusionDiffEditPipeline`] لترميز النص. ستستخدم مشفر النص لحساب التضمينات النصية:

```py
import torch
from diffusers import StableDiffusionDiffEditPipeline

pipeline = StableDiffusionDiffEditPipeline.from_pretrained(
"stabilityai/stable-diffusion-2-1"، torch_dtype=torch.float16، use_safetensors=True
)
pipeline.enable_model_cpu_offload()
pipeline.enable_vae_slicing()

@torch.no_grad()
def embed_prompts(sentences, tokenizer, text_encoder, device="cuda"):
embeddings = []
for sent in sentences:
text_inputs = tokenizer(
sent،
padding="max_length"،
max_length=tokenizer.model_max_length،
truncation=True،
return_tensors="pt"،
)
text_input_ids = text_inputs.input_ids
prompt_embeds = text_encoder(text_input_ids.to(device)، attention_mask=None)[0]
embeddings.append(prompt_embeds)
return torch.concatenate(embeddings, dim=0).mean(dim=0).unsqueeze(0)

source_embeds = embed_prompts(source_prompts, pipeline.tokenizer, pipeline.text_encoder)
target_embeds = embed_prompts(target_prompts, pipeline.tokenizer, pipeline.text_encoder)
```

أخيرًا، قم بتمرير التضمينات إلى الدالتين [`~StableDiffusionDiffEditPipeline.generate_mask`] و [`~StableDiffusionDiffEditPipeline.invert`]`، والأنبوب لإنشاء الصورة:

```diff
from diffusers import DDIMInverseScheduler, DDIMScheduler
from diffusers.utils import load_image, make_image_grid
from PIL import Image

pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)

img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"
raw_image = load_image(img_url).resize((768, 768))

mask_image = pipeline.generate_mask(
image=raw_image،
-     source_prompt=source_prompt،
-     target_prompt=target_prompt،
+     source_prompt_embeds=source_embeds،
+     target_prompt_embeds=target_embeds،
)

inv_latents = pipeline.invert(
-     prompt=source_prompt،
+     prompt_embeds=source_embeds،
image=raw_image،
).latents

output_image = pipeline(
mask_image=mask_image،
image_latents=inv_latents،
-     prompt=target_prompt،
-     negative_prompt=source_prompt،
+     prompt_embeds=target_embeds،
+     negative_prompt_embeds=source_embeds،
).images[0]
mask_image = Image.fromarray((mask_image.squeeze()*255).astype("uint8")، "L")
make_image_grid([raw_image، mask_image، output_image]، rows=1، cols=3)
```
## إنشاء عنوان توضيحي للانعكاس

على الرغم من أنه يمكنك استخدام `source_prompt` كعنوان توضيحي للمساعدة في إنشاء الانحرافات الجزئية، إلا أنه يمكنك أيضًا استخدام نموذج [BLIP](https://huggingface.co/docs/transformers/model_doc/blip) لتوليد عنوان توضيحي تلقائيًا.

قم بتحميل نموذج BLIP ومعالج BLIP من مكتبة 🤗 Transformers:

```py
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16, low_cpu_mem_usage=True)
```

قم بإنشاء دالة مساعدة لتوليد عنوان توضيحي من صورة الإدخال:

```py
@torch.no_grad()
def generate_caption(images, caption_generator, caption_processor):
    text = "a photograph of"

    inputs = caption_processor(images, text, return_tensors="pt").to(device="cuda", dtype=caption_generator.dtype)
    caption_generator.to("cuda")
    outputs = caption_generator.generate(**inputs, max_new_tokens=128)

    # offload caption generator
    caption_generator.to("cpu")

    caption = caption_processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return caption
```

قم بتحميل صورة إدخال وقم بتوليد عنوان توضيحي لها باستخدام دالة `generate_caption`:

```py
from diffusers.utils import load_image

img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"
raw_image = load_image(img_url).resize((768, 768))
caption = generate_caption(raw_image, model, processor)
```

<div class="flex justify-center">
<figure>
<img class="rounded-xl" src="https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"/>
<figcaption class="text-center">العنوان المولد: "صورة فوتوغرافية لفاكهة في وعاء على طاولة"</figcaption>
</figure>
</div>

الآن، يمكنك وضع العنوان التوضيحي في دالة [`~StableDiffusionDiffEditPipeline.invert`] لإنشاء الانحرافات الجزئية!