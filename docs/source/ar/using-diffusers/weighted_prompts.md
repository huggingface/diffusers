# تقنيات المحفزات

[[open-in-colab]]

المحفزات مهمة لأنها تصف ما تريد من نموذج الانتشار توليده. أفضل المحفزات تكون مفصلة ومحددة وجيدة التنظيم لمساعدة النموذج على تحقيق رؤيتك. ولكن صياغة محفز رائع يستغرق وقتًا وجهدًا، وفي بعض الأحيان قد لا يكون ذلك كافيًا لأن اللغة والكلمات قد لا تكون دقيقة. هنا تحتاج إلى تعزيز محفزك بتقنيات أخرى، مثل تحسين المحفزات ووزن المحفزات، للحصول على النتائج التي تريدها.

سيوضح هذا الدليل كيفية استخدام تقنيات المحفزات هذه لتوليد صور عالية الجودة بمجهود أقل وتعديل وزن كلمات رئيسية معينة في محفز.

## هندسة المحفزات

> [!TIP]
> هذا ليس دليلًا شاملاً عن هندسة المحفزات، ولكنه سيساعدك على فهم الأجزاء الضرورية لمحفز جيد. نشجعك على الاستمرار في تجربة محفزات مختلفة ودمجها بطرق جديدة لمعرفة ما يناسبك. كلما كتبت محفزات أكثر، ستطور حدسًا لما ينجح وما لا ينجح!

تؤدي النماذج الجديدة للانتشار وظيفة جيدة في توليد صور عالية الجودة من محفز أساسي، ولكن لا يزال من المهم إنشاء محفز جيد الصياغة للحصول على أفضل النتائج. فيما يلي بعض النصائح لكتابة محفز جيد:

1. ما هي صيغة الصورة؟ هل هي صورة فوتوغرافية أم لوحة زيتية أم توضيح ثلاثي الأبعاد أم شيء آخر؟
2. ما هو موضوع الصورة؟ هل هو شخص أو حيوان أو كائن أو مشهد؟
3. ما هي التفاصيل التي تريد رؤيتها في الصورة؟ هنا يمكنك أن تكون مبدعًا حقًا والاستمتاع بتجربة كلمات مختلفة لإضفاء الحياة على صورتك. على سبيل المثال، ما هو الإضاءة؟ ما هي الحالة المزاجية والجمالية؟ ما هو أسلوب الفن أو التوضيح الذي تبحث عنه؟ كلما كانت الكلمات التي تستخدمها أكثر تحديدًا ودقة، كان فهم النموذج لما تريد توليده أفضل.

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/plain-prompt.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">"صورة فوتوغرافية لأريكة على شكل موزة في غرفة المعيشة"</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/detail-prompt.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">"أريكة على شكل موزة صفراء زاهية تجلس في غرفة معيشة مريحة، وتلتف منحنيتها حول كومة من الوسائد الملونة. وعلى الأرض الخشبية، تضيف سجادة ذات نقوش لمسة من السحر غير المتكلف، وتقف نبتة في الزاوية، ممتدة نحو أشعة الشمس التي تتسلل من النوافذ"</figcaption>
</div>
</div>

## تحسين المحفزات باستخدام GPT2

تحسين المحفزات هي تقنية لتحسين جودة المحفز بسرعة دون بذل الكثير من الجهد في بنائه. تستخدم هذه التقنية نموذجًا مثل GPT2 الذي تم تدريبه مسبقًا على محفزات نص Stable Diffusion لاستكمال المحفز تلقائيًا بكلمات رئيسية إضافية مهمة لتوليد صور عالية الجودة.

تعمل هذه التقنية من خلال تجميع قائمة بكلمات رئيسية محددة وإجبار النموذج على توليد تلك الكلمات لتعزيز المحفز الأصلي. بهذه الطريقة، يمكن أن يكون محفزك "قطة"، ويمكن لـ GPT2 تعزيز المحفز إلى "لقطة سينمائية لفيلم لقطة لقطة لقطة في الشمس على سطح في تركيا، مفصلة للغاية، وميزانية ضخمة لفيلم هوليوود، سينماسكوب، مزاجي، ملحمي، جميل، حبيبات الفيلم جودة التركيز الحاد جميلة مفصلة معقدة مذهلة ملحمية".

> [!TIP]
> يجب عليك أيضًا استخدام [ضجيج التعويض](https://www.crosslabs.org//blog/diffusion-with-offset-noise) LoRA لتحسين التباين في الصور الفاتحة والداكنة وإنشاء إضاءة أفضل بشكل عام. متاح هذا [LoRA](https://hf.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_offset_example-lora_1.0.safetensors) من [stabilityai/stable-diffusion-xl-base-1.0](https://hf.co/stabilityai/stable-diffusion-xl-base-1.0).

ابدأ بتحديد بعض الأساليب وقائمة من الكلمات (يمكنك الاطلاع على قائمة أكثر شمولاً من [الكلمات](https://hf.co/LykosAI/GPT-Prompt-Expansion-Fooocus-v2/blob/main/positive.txt) و [الأساليب](https://github.com/lllyasviel/Fooocus/tree/main/sdxl_styles) التي يستخدمها Fooocus) لتعزيز محفز بها.

```py
import torch
from transformers import GenerationConfig, GPT2LMHeadModel, GPT2Tokenizer, LogitsProcessor, LogitsProcessorList
from diffusers import StableDiffusionXLPipeline

styles = {
"cinematic": "cinematic film still of {prompt}, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain",
"anime": "anime artwork of {prompt}, anime style, key visual, vibrant, studio anime, highly detailed",
"photographic": "cinematic photo of {prompt}, 35mm photograph, film, professional, 4k, highly detailed",
"comic": "comic of {prompt}, graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
"lineart": "line art drawing {prompt}, professional, sleek, modern, minimalist, graphic, line art, vector graphics",
"pixelart": " pixel-art {prompt}, low-res, blocky, pixel art style, 8-bit graphics",
}

words = [
"aesthetic", "astonishing", "beautiful", "breathtaking", "composition", "contrasted", "epic", "moody", "enhanced",
"exceptional", "fascinating", "flawless", "glamorous", "glorious", "illumination", "impressive", "improved",
"inspirational", "magnificent", "majestic", "hyperrealistic", "smooth", "sharp", "focus", "stunning", "detailed",
"intricate", "dramatic", "high", "quality", "perfect", "light", "ultra", "highly", "radiant", "satisfying",
"soothing", "sophisticated", "stylish", "sublime", "terrific", "touching", "timeless", "wonderful", "unbelievable",
"elegant", "awesome", "amazing", "dynamic", "trendy",
]
```

قد تكون لاحظت في قائمة `words`، أن هناك كلمات معينة يمكن اقترانها لخلق شيء أكثر معنى. على سبيل المثال، يمكن دمج كلمتي "high" و "quality" لتصبح "high quality". دعنا نقرن هذه الكلمات معًا ونزيل الكلمات التي لا يمكن اقترانها.

```py
word_pairs = ["highly detailed", "high quality", "enhanced quality", "perfect composition", "dynamic light"]

def find_and_order_pairs(s, pairs):
words = s.split()
found_pairs = []
for pair in pairs:
pair_words = pair.split()
if pair_words[0] in words and pair_words[1] in words:
found_pairs.append(pair)
words.remove(pair_words[0])
words.remove(pair_words[1])

for word in words[:]:
for pair in pairs:
if word in pair.split():
words.remove(word)
break
ordered_pairs = ", ".join(found_pairs)
remaining_s = ", ".join(words)
return ordered_pairs, remaining_s
```

بعد ذلك، قم بتنفيذ فئة [`~transformers.LogitsProcessor` مخصصة](https://huggingface.co/transformers/main_classes/logits_processor.html) تقوم بتعيين الرموز في قائمة `words` بقيمة 0 وتعيين الرموز غير الموجودة في قائمة `words` بقيمة سالبة حتى لا يتم اختيارها أثناء التوليد. بهذه الطريقة، يكون التوليد متحيزًا نحو الكلمات الموجودة في قائمة `words`. بعد استخدام كلمة من القائمة، يتم أيضًا تعيينها بقيمة سالبة حتى لا يتم اختيارها مرة أخرى.

```py
class CustomLogitsProcessor(LogitsProcessor):
def __init__(self, bias):
super().__init__()
self.bias = bias

def __call__(self, input_ids, scores):
if len(input_ids.shape) == 2:
last_token_id = input_ids[0, -1]
self.bias[last_token_id] = -1e10
return scores + self.bias

word_ids = [tokenizer.encode(word, add_prefix_space=True)[0] for word in words]
bias = torch.full((tokenizer.vocab_size,), -float("Inf")).to("cuda")
bias[word_ids] = 0
processor = CustomLogitsProcessor(bias)
processor_list = LogitsProcessorList([processor])
```

قم بدمج المحفز ومحفز `cinematic` المحدد في قاموس `styles` سابقًا.

```py
prompt = "a cat basking in the sun on a roof in Turkey"
style = "cinematic"

prompt = styles[style].format(prompt=prompt)
prompt
"cinematic film still of a cat basking in the sun on a roof in Turkey, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain"
```

قم بتحميل برنامج تعليم GPT2 ونموذج من نقطة التحقق [Gustavosta/MagicPrompt-Stable-Diffusion](https://huggingface.co/Gustavosta/MagicPrompt-Stable-Diffusion) (تم تدريب نقطة التحقق هذه خصيصًا لتوليد المحفزات) لتعزيز المحفز.

```py
tokenizer = GPT2Tokenizer.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
model = GPT2LMHeadModel.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion", torch_dtype=torch.float16).to(
"cuda"
)
model.eval()

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
token_count = inputs["input_ids"].shape[1]
max_new_tokens = 50 - token_count

generation_config = GenerationConfig(
penalty_alpha=0.7,
top_k=50,
eos_token_id=model.config.eos_token_id,
pad_token_id=model.config.eos_token_id,
pad_token=model.config.pad_token_id,
do_sample=True,
)

with torch.no_grad():
generated_ids = model.generate(
input_ids=inputs["input_ids"],
attention_mask=inputs["attention_mask"],
max_new_tokens=max_new_tokens,
generation_config=generation_config,
logits_processor=proccesor_list,
)
```

بعد ذلك، يمكنك دمج المحفز المدخل والمحفز المولد. لا تتردد في إلقاء نظرة على المحفز المولد (`generated_part`)، وأزواج الكلمات التي تم العثور عليها (`pairs`)، والكلمات المتبقية (`words`). كل هذا مضمن في `enhanced_prompt`.

```py
output_tokens = [tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_ids]
input_part, generated_part = output_tokens[0][: len(prompt)], output_tokens[0][len(prompt) :]
pairs, words = find_and_order_pairs(generated_part, word_pairs)
formatted_generated_part = pairs + ", " + words
enhanced_prompt = input_part + ", " + formatted_generated_part
enhanced_prompt
["cinematic film still of a cat basking in the sun on a roof in Turkey, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain quality sharp focus beautiful detailed intricate stunning amazing epic"]
```

أخيرًا، قم بتحميل خط أنابيب ووزن LoRA لضجيج التعويض بوزن *منخفض* لتوليد صورة باستخدام المحفز المعزز.

```py
pipeline = StableDiffusionXLPipeline.from_pretrained(
"RunDiffusion/Juggernaut-XL-v9", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

pipeline.load_lora_weights(
"stabilityai/stable-diffusion-xl-base-1.0",
weight_name="sd_xl_offset_example-lora_1.0.safetensors",
adapter_name="offset",
)
pipeline.set_adapters(["offset"], adapter_weights=[0.2])

image = pipeline(
enhanced_prompt,
width=1152,
height=896,
guidance_scale=7.5,
num_inference_steps=25,
).images[0]
image
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/non-enhanced-prompt.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">"قطة تستلقي في الشمس على سطح في تركيا"</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/enhanced-prompt.png"/>
<figcaption class="mt-₂ text-center text-sm text-gray-500">"لقطة سينمائية لفيلم لقطة لقطة لقطة في الشمس على سطح في تركيا، مفصلة للغاية، وميزانية ضخمة لفيلم هوليوود، سينماسكوب، مزاجي، ملحمي، جميل، حبيبات الفيلم جودة التركيز الحاد جميلة مفصلة معقدة مذهلة ملحمية"</figcaption>
</div>
</div>
## وزن المطالبة

يوفر وزن المطالبة طريقة لتأكيد أو تقليل أهمية أجزاء معينة من مطالبة ما، مما يتيح مزيدًا من التحكم في الصورة المولدة. يمكن أن تتضمن المطالبة عدة مفاهيم، والتي يتم تحويلها إلى تضمينات نصية سياقية. تستخدم النماذج التضمينات لتهيئة طبقات الاهتمام المتقاطع الخاصة بها لتوليد صورة (اقرأ منشور المدونة Stable Diffusion [1] لمعرفة المزيد حول كيفية عملها).

يعمل وزن المطالبة عن طريق زيادة أو تقليل مقياس متجه التضمين النصي الذي يقابله في المطالبة. قد لا ترغب في أن يركز النموذج على جميع المفاهيم بالتساوي. وأبسط طريقة لإعداد التضمينات ذات الأهمية النسبية هي استخدام [Compel] [2]، وهو مكتبة لوزن المطالبة النصية ودمجها. بمجرد حصولك على التضمينات ذات الأهمية النسبية، يمكنك تمريرها إلى أي خط أنابيب يحتوي على وسيط [`prompt_embeds`] [3] (واختياريًا [`negative_prompt_embeds`] [4])، مثل [`StableDiffusionPipeline`] [5]، و [`StableDiffusionControlNetPipeline`] [6]، و [`StableDiffusionXLPipeline`] [7].

<Tip>

إذا لم يكن خط الأنابيب المفضل لديك يحتوي على وسيط `prompt_embeds`، فيرجى فتح [قضية] [8] حتى نتمكن من إضافته!

</Tip>

سيوضح هذا الدليل كيفية وزن ودمج مطالباتك باستخدام Compel في 🤗 Diffusers.

قبل البدء، تأكد من أن لديك أحدث إصدار من Compel مثبتًا:

```py
# قم بإلغاء التعليق لتثبيته في Colab
#! pip install compel --upgrade
```

بالنسبة لهذا الدليل، دعنا نقوم بتوليد صورة باستخدام المطالبة "قطة حمراء تلعب بكرة" باستخدام [`StableDiffusionPipeline`] :

```py
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_safetensors=True)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

prompt = "a red cat playing with a ball"

generator = torch.Generator(device="cpu").manual_seed(33)

image = pipe(prompt, generator=generator, num_inference_steps=20).images[0]
image
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/compel/forest_0.png"/>
</div>

### الوزن

ستلاحظ أنه لا توجد "كرة" في الصورة! دعنا نستخدم Compel لزيادة وزن مفهوم "كرة" في المطالبة. قم بإنشاء كائن [`Compel`] [9]، ومرر محدد مواقع ومعالج نصي له:

```py
from compel import Compel

compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
```

يستخدم Compel `+` أو `-` لزيادة أو تقليل وزن كلمة في المطالبة. لزيادة وزن "كرة":

<Tip>

`+` يقابله القيمة `1.1`، `++` يقابله `1.1^2`، وهكذا. وبالمثل، يقابل `-` القيمة `0.9` ويقابل `--` القيمة `0.9^2`. لا تتردد في التجربة عن طريق إضافة المزيد من `+` أو `-` في مطالبتك!

</Tip>

```py
prompt = "a red cat playing with a ball++"
```

مرر المطالبة إلى `compel_proc` لإنشاء التضمينات الجديدة التي تتمتع بأهمية نسبية والتي يتم تمريرها إلى خط الأنابيب:

```py
prompt_embeds = compel_proc(prompt)
generator = torch.manual_seed(33)

image = pipe(prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=20).images[0]
image
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/compel/forest_1.png"/>
</div>

لتخفيض وزن أجزاء من المطالبة، استخدم اللاحقة `-`:

```py
prompt = "a red------- cat playing with a ball"
prompt_embeds = compel_proc(prompt)

generator = torch.manual_seed(33)

image = pipe(prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=20).images[0]
image
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-neg.png"/>
</div>

يمكنك حتى زيادة أو تقليل وزن مفاهيم متعددة في نفس المطالبة:

```py
prompt = "a red cat++ playing with a ball----"
prompt_embeds = compel_proc(prompt)

generator = torch.manual_seed(33)

image = pipe(prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=20).images[0]
image
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-pos-neg.png"/>
</div>

### المزج

يمكنك أيضًا إنشاء مزيج مرجح من المطالبات عن طريق إضافة `.blend()` إلى قائمة من المطالبات وتمرير بعض الأوزان لها. قد لا ينتج مزيجك دائمًا النتيجة التي تتوقعها لأنه يكسر بعض الافتراضات حول كيفية عمل المشفر النصي، لذا قم بالتجربة والاستمتاع به!

```py
prompt_embeds = compel_proc('("a red cat playing with a ball", "jungle").blend(0.7, 0.8)')
generator = torch.Generator(device="cuda").manual_seed(33)

image = pipe(prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=20).images[0]
image
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-blend.png"/>
</div>

### العطف

يؤدي العطف إلى نشر كل مطالبة بشكل مستقل ويقوم بدمج نتائجها بواسطة مجموعها المرجح. أضف `.and()` في نهاية قائمة من المطالبات لإنشاء عطف:

```py
prompt_embeds = compel_proc('["a red cat", "playing with a", "ball"].and()')
generator = torch.Generator(device="cuda").manual_seed(55)

image = pipe(prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=20).images[0]
image
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-conj.png"/>
</div>

### الانقلاب النصي

[الانقلاب النصي] [10] هو تقنية لتعلم مفهوم محدد من بعض الصور التي يمكنك استخدامها لتوليد صور جديدة مشروطة بذلك المفهوم.

قم بإنشاء خط أنابيب واستخدم دالة [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] [11] لتحميل التضمينات الانقلابية النصية (لا تتردد في تصفح [Stable Diffusion Conceptualizer] [12] لأكثر من 100 مفهوم مدرب):

```py
import torch
from diffusers import StableDiffusionPipeline
from compel import Compel, DiffusersTextualInversionManager

pipe = StableDiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,
use_safetensors=True, variant="fp16").to("cuda")
pipe.load_textual_inversion("sd-concepts-library/midjourney-style")
```

يوفر Compel فئة `DiffusersTextualInversionManager` لتبسيط وزن المطالبة باستخدام الانقلاب النصي. قم بتنفيذ `DiffusersTextualInversionManager` ومرره إلى فئة `Compel`:

```py
textual_inversion_manager = DiffusersTextualInversionManager(pipe)
compel_proc = Compel(
tokenizer=pipe.tokenizer,
text_encoder=pipe.text_encoder,
textual_inversion_manager=textual_inversion_manager)
```

قم بدمج المفهوم لتهيئة مطالبة باستخدام بناء الجملة `<concept>`:

```py
prompt_embeds = compel_proc('("A red cat++ playing with a ball <midjourney-style>")')

image = pipe(prompt_embeds=prompt_embeds).images[0]
image
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-text-inversion.png"/>
</div>

### DreamBooth

[DreamBooth] [13] هي تقنية لتوليد صور سياقية لموضوع ما بناءً على عدد قليل فقط من الصور الخاصة بالموضوع للتدريب عليها. إنه مشابه للانقلاب النصي، ولكن DreamBooth يقوم بتدريب النموذج بالكامل في حين أن الانقلاب النصي يقوم فقط بتعديل التضمينات النصية. وهذا يعني أنه يجب عليك استخدام [`~DiffusionPipeline.from_pretrained`] [14] لتحميل نموذج DreamBooth (لا تتردد في تصفح [Stable Diffusion Dreambooth Concepts Library] [15] لأكثر من 100 نموذج مدرب):

```py
import torch
from diffusers import DiffusionPipeline, UniPCMultistepScheduler
from compel import Compel

pipe = DiffusionPipeline.from_pretrained("sd-dreambooth-library/dndcoverart-v1", torch_dtype=torch.float16).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
```

قم بإنشاء فئة `Compel` مع محدد مواقع ومشفر نصي، ومرر مطالبتك إليها. اعتمادًا على النموذج الذي تستخدمه، ستحتاج إلى دمج المعرف الفريد للنموذج في مطالبتك. على سبيل المثال، يستخدم نموذج `dndcoverart-v1` المعرف `dndcoverart`:

```py
compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
prompt_embeds = compel_proc('("magazine cover of a dndcoverart dragon, high quality, intricate details, larry elmore art style").and()')
image = pipe(prompt_embeds=prompt_embeds).images[0]
image
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-dreambooth.png"/>
</div>

### Stable Diffusion XL

يحتوي Stable Diffusion XL (SDXL) على محددي مواقع ومشفرين نصيين، لذا فإن استخدامه مختلف بعض الشيء. لمعالجة ذلك، يجب تمرير كلا المحددين والمشفرين إلى فئة `Compel`:

```py
from compel import Compel, ReturnedEmbeddingsType
from diffusers import DiffusionPipeline
from diffusers.utils import make_image_grid
import torch

pipeline = DiffusionPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
variant="fp16",
use_safetensors=True,
torch_dtype=torch.float16
).to("cuda")

compel = Compel(
tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
requires_pooled=[False, True]
)
```

هذه المرة، دعنا نزيد وزن "كرة" بمعامل 1.5 للمطالبة الأولى، ونقلل وزن "كرة" إلى 0.6 للمطالبة الثانية. يتطلب [`StableDiffusionXLPipeline`] [16] أيضًا [`pooled_prompt_embeds`] [17] (واختياريًا [`negative_pooled_prompt_embeds`] [18])، لذا يجب عليك تمريرها إلى خط الأنابيب جنبًا إلى جنب مع التوتنات الشرطية:

```py
# تطبيق الأوزان
prompt = ["a red cat playing with a (ball)1.5", "a red cat playing with a (ball)0.6"]
conditioning, pooled = compel(prompt)

# توليد الصورة
generator = [torch.Generator().manual_seed(33) for _ in range(len(prompt))]
images = pipeline(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, generator=generator, num_inference_steps=30).images
make_image_grid(images, rows=1, cols=2)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/compel/sdxl_ball1.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">"a red cat playing with a (ball)1.5"</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/compel/sdxl_ball2.png"/>
<figcaption class="mt-۲ text-center text-sm text-gray-500">"a red cat playing with a (ball)0.6"</figcaption>
</div>
</div>