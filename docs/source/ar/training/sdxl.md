# Stable Diffusion XL

<Tip warning={true}>
هذا النص البرمجي تجريبي، ومن السهل أن ينحرف عن المسار الصحيح وأن يواجه مشكلات مثل النسيان الكارثي. جرّب استكشاف مختلف البارامترات للحصول على أفضل النتائج لمجموعة بياناتك.
</Tip>

[Stable Diffusion XL (SDXL)](https://hf.co/papers/2307.01952) هو إصدار أكبر وأكثر قوة من نموذج Stable Diffusion، وقادر على إنتاج صور ذات دقة أعلى.

إن شبكة UNet في SDXL أكبر بثلاث مرات، ويضيف النموذج مشفر نص ثانٍ إلى البنية. واعتمادًا على الأجهزة المتوفرة لديك، يمكن أن يكون هذا مكثفًا جدًا من الناحية الحسابية وقد لا يعمل على وحدة معالجة رسومات (GPU) للمستهلك مثل Tesla T4. للمساعدة في تكييف هذا النموذج الأكبر في الذاكرة ولتسريع التدريب، جرّب تمكين `gradient_checkpointing`، و`mixed_precision`، و`gradient_accumulation_steps`. يمكنك تقليل استخدام الذاكرة لديك أكثر من خلال تمكين الاهتمام الفعال للذاكرة باستخدام [xFormers](../optimization/xformers) واستخدام محسّن 8-bit من [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).

سيتناول هذا الدليل نص البرنامج النصي للتدريب [train_text_to_image_sdxl.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_sdxl.py) لمساعدتك على التعرف عليه بشكل أفضل، وكيف يمكنك تكييفه مع حالتك الاستخدامية الخاصة.

قبل تشغيل النص البرمجي، تأكد من تثبيت المكتبة من المصدر:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

بعد ذلك، انتقل إلى مجلد الأمثلة الذي يحتوي على نص البرنامج النصي للتدريب وقم بتثبيت التبعيات المطلوبة للنص البرمجي الذي تستخدمه:

```bash
cd examples/text_to_image
pip install -r requirements_sdxl.txt
```

<Tip>
🤗 Accelerate هي مكتبة للمساعدة في التدريب على وحدات معالجة رسومات (GPU) / وحدات معالجة الرسومات (TPU) متعددة أو باستخدام الدقة المختلطة. سيقوم تلقائيًا بتكوين إعداد التدريب الخاص بك بناءً على الأجهزة وبيئتك. اطلع على الجولة السريعة لـ 🤗 Accelerate [Quick tour](https://huggingface.co/docs/accelerate/quicktour) لمعرفة المزيد.
</Tip>

قم بتهيئة بيئة 🤗 Accelerate:

```bash
accelerate config
```

لإعداد بيئة 🤗 Accelerate الافتراضية دون اختيار أي تكوينات:

```bash
accelerate config default
```

أو إذا لم يدعم بيئتك غلافًا تفاعليًا، مثل دفتر الملاحظات، فيمكنك استخدام ما يلي:

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

أخيرًا، إذا كنت تريد تدريب نموذج على مجموعة البيانات الخاصة بك، فراجع دليل [إنشاء مجموعة بيانات للتدريب](create_dataset) لمعرفة كيفية إنشاء مجموعة بيانات تعمل مع نص البرنامج النصي للتدريب.

## معلمات النص البرمجي

<Tip>
تسلط الأقسام التالية الضوء على أجزاء من نص البرنامج النصي للتدريب المهمة لفهم كيفية تعديلها، ولكنها لا تغطي كل جانب من جوانب النص البرمجي بالتفصيل. إذا كنت مهتمًا بمعرفة المزيد، فلا تتردد في قراءة [النص البرمجي](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/text_to_image/train_text_to_image_sdxl.py) وأخبرنا إذا كان لديك أي أسئلة أو مخاوف.
</Tip>

يوفر نص البرنامج النصي للتدريب العديد من المعلمات لمساعدتك في تخصيص عملية التدريب. توجد جميع المعلمات ووصفاتها في دالة [`parse_args()`](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/text_to_image/train_text_to_image_sdxl.py#L129). توفر هذه الدالة قيم افتراضية لكل معلمة، مثل حجم دفعة التدريب ومعدل التعلم، ولكن يمكنك أيضًا تعيين قيمك الخاصة في أمر التدريب إذا أردت ذلك.

على سبيل المثال، لتسريع التدريب باستخدام الدقة المختلطة بتنسيق bf16، أضف معلمة `--mixed_precision` إلى أمر التدريب:

```bash
accelerate launch train_text_to_image_sdxl.py \
--mixed_precision="bf16"
```

تتشابه معظم المعلمات مع معلمات دليل التدريب [Text-to-image](text2image#script-parameters)، لذلك ستركز على المعلمات ذات الصلة بتدريب SDXL في هذا الدليل.

- `--pretrained_vae_model_name_or_path`: المسار إلى VAE مُدرب مسبقًا؛ تُعرف VAE الخاصة بـ SDXL بأنها تعاني من عدم استقرار رقمي، لذلك تسمح هذه المعلمة بتحديد VAE أفضل [VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)

- `--proportion_empty_prompts`: نسبة موجهات الصور التي سيتم استبدالها بسلسلة فارغة

- `--timestep_bias_strategy`: المكان (في وقت سابق مقابل لاحق) في الخطوة الزمنية لتطبيق الانحياز، والذي يمكن أن يشجع النموذج على تعلم تفاصيل التردد المنخفض أو العالي

- `--timestep_bias_multiplier`: وزن الانحياز لتطبيقه على الخطوة الزمنية

- `--timestep_bias_begin`: الخطوة الزمنية لبدء تطبيق الانحياز

- `--timestep_bias_end`: الخطوة الزمنية لإنهاء تطبيق الانحياز

- `--timestep_bias_portion`: نسبة الخطوات الزمنية لتطبيق الانحياز عليها

 ### وزن Min-SNR

يمكن أن تساعد استراتيجية الوزن Min-SNR في التدريب من خلال إعادة توازن الخسارة لتحقيق تقارب أسرع. يدعم نص البرنامج النصي للتدريب التنبؤ إما بـ `epsilon` (الضوضاء) أو `v_prediction`، ولكن Min-SNR متوافق مع كلا نوعي التنبؤ. استراتيجية الترجيح هذه مدعومة فقط بواسطة PyTorch وغير متوفرة في نص البرنامج النصي للتدريب Flax.

أضف معلمة `--snr_gamma` وقم بتعيينها على القيمة الموصى بها 5.0:

```bash
accelerate launch train_text_to_image_sdxl.py \
--snr_gamma=5.0
```

## نص البرنامج النصي للتدريب

نص البرنامج النصي للتدريب مشابه أيضًا لنص البرنامج النصي للتدريب في دليل [Text-to-image](text2image#training-script)، ولكنه تم تعديله لدعم التدريب على SDXL. سيركز هذا الدليل على التعليمات البرمجية الفريدة لنص البرنامج النصي للتدريب على SDXL.

يبدأ بإنشاء دالات ل[رموز المميزة للمحثات](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/text_to_image/train_text_to_image_sdxl.py#L478) لحساب تضمين المحث، ولحساب تضمين الصورة باستخدام [VAE](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/text_to_image/train_text_to_image_sdxl.py#L519). بعد ذلك، سترى دالة لإنشاء [أوزان الخطوات الزمنية](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/text_to_image/train_text_to_image_sdxl.py#L531) اعتمادًا على عدد الخطوات الزمنية واستراتيجية الانحياز للخطوة الزمنية التي سيتم تطبيقها.

داخل دالة [`main()`](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/text_to_image/train_text_to_image_sdxl.py#L572)، بالإضافة إلى تحميل محول الرموز، يقوم النص البرمجي بتحميل محول رموز ثانٍ ومشفر نص ثانٍ لأن بنية SDXL تستخدم اثنين من كل منهما:

```py
tokenizer_one = AutoTokenizer.from_pretrained(
args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False
)
tokenizer_two = AutoTokenizer.from_pretrained(
args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, use_fast=False
)

text_encoder_cls_one = import_model_class_from_model_name_or_path(
args.pretrained_model_name_or_path, args.revision
)
text_encoder_cls_two = import_model_class_from_model_name_or_path(
args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
)
```

يتم حساب [تضمين المحثات والصور](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/text_to_image/train_text_to_image_sdxl.py#L857) أولاً ويتم الاحتفاظ بها في الذاكرة، وهو ما لا يمثل مشكلة عادةً لمجموعة بيانات أصغر، ولكن بالنسبة لمجموعات البيانات الأكبر، يمكن أن يؤدي ذلك إلى مشكلات في الذاكرة. إذا كان الأمر كذلك، فيجب عليك حفظ التضمينات المحسوبة مسبقًا على القرص بشكل منفصل وتحميلها في الذاكرة أثناء عملية التدريب (راجع طلب السحب هذا [PR](https://github.com/huggingface/diffusers/pull/4505) لمزيد من المناقشة حول هذا الموضوع).

```py
text_encoders = [text_encoder_one, text_encoder_two]
tokenizers = [tokenizer_one, tokenizer_two]
compute_embeddings_fn = functools.partial(
encode_prompt,
text_encoders=text_encoders,
tokenizers=tokenizers,
proportion_empty_prompts=args.proportion_empty_prompts,
caption_column=args.caption_column,
)

train_dataset = train_dataset.map(compute_embeddings_fn, batched=True, new_fingerprint=new_fingerprint)
train_dataset = train_dataset.map(
compute_vae_encodings_fn,
batched=True,
batch_size=args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps,
new_fingerprint=new_fingerprint_for_vae,
)
```

بعد حساب التضمينات، يتم حذف مشفر النص وVAE ومحول الرموز لتحرير بعض الذاكرة:

```py
del text_encoders, tokenizers, vae
gc.collect()
torch.cuda.empty_cache()
```

أخيرًا، تتولى حلقة التدريب [training loop](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/text_to_image/train_text_to_image_sdxl.py#L943) بقية العملية. إذا اخترت تطبيق استراتيجية انحياز الخطوة الزمنية، فسترى أن أوزان الخطوات الزمنية يتم حسابها وإضافتها كضوضاء:

```py
weights = generate_timestep_weights(args, noise_scheduler.config.num_train_timesteps).to(
model_input.device
)
timesteps = torch.multinomial(weights, bsz, replacement=True).long()

noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
```

إذا كنت تريد معرفة المزيد حول كيفية عمل حلقة التدريب، فراجع البرنامج التعليمي [Understanding pipelines, models and schedulers](../using-diffusers/write_own_pipeline) الذي يكسر النمط الأساسي لعملية إزالة الضوضاء.
## تشغيل السكربت

عندما تنتهي من إجراء جميع التغييرات أو تكون راضيا عن التهيئة الافتراضية، ستكون جاهزًا لتشغيل سكربت التدريب! 🚀

دعونا نتدرب على مجموعة بيانات [Naruto BLIP captions](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) لإنشاء شخصيات ناروتو الخاصة بك. قم بتعيين متغيرات البيئة `MODEL_NAME` و `DATASET_NAME` إلى النموذج ومجموعة البيانات (إما من Hub أو مسار محلي). يجب عليك أيضًا تحديد VAE بخلاف SDXL VAE (إما من Hub أو مسار محلي) باستخدام `VAE_NAME` لتجنب عدم الاستقرار العددي.

<Tip>

لمراقبة تقدم التدريب باستخدام Weights & Biases، أضف المعامل `--report_to=wandb` إلى أمر التدريب. ستحتاج أيضًا إلى إضافة `--validation_prompt` و `--validation_epochs` إلى أمر التدريب لتتبع النتائج. يمكن أن يكون هذا مفيدًا جدًا لتصحيح أخطاء النموذج وعرض النتائج الوسيطة.

</Tip>

```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch train_text_to_image_sdxl.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--pretrained_vae_model_Multiplier_path=$VAE_NAME \
--dataset_name=$DATASET_NAME \
--enable_xformers_memory_efficient_attention \
--resolution=512 \
--center_crop \
--random_flip \
--proportion_empty_prompts=0.2 \
--train_batch_size=1 \
--gradient_accumulation_steps=4 \
--gradient_checkpointing \
--max_train_steps=10000 \
--use_8bit_adam \
--learning_rate=1e-06 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--mixed_precision="fp16" \
--report_to="wandb" \
--validation_prompt="a cute Sundar Pichai creature" \
--validation_epochs 5 \
--checkpointing_steps=5000 \
--output_dir="sdxl-naruto-model" \
--push_to_hub
```

بعد الانتهاء من التدريب، يمكنك استخدام نموذج SDXL المدرب حديثًا للاستنتاج!

<hfoptions id="inference">

<hfoption id="PyTorch">

```py
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("path/to/your/model", torch_dtype=torch.float16).to("cuda")

prompt = "A naruto with green eyes and red legs."
image = pipeline(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("naruto.png")
```

</hfoption>

<hfoption id="PyTorch XLA">

[PyTorch XLA](https://pytorch.org/xla) يسمح لك بتشغيل PyTorch على أجهزة XLA مثل TPUs، والتي يمكن أن تكون أسرع. خطوة التسخين الأولية تستغرق وقتًا أطول لأن النموذج يحتاج إلى التجميع والتحسين. ومع ذلك، فإن الاستدعاءات اللاحقة لخط الأنابيب على إدخال **بنفس طول** الفكرة الأصلية أسرع بكثير لأنه يمكنه إعادة استخدام الرسم البياني المحسن.

```py
from diffusers import DiffusionPipeline
import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0").to(device)

prompt = "A naruto with green eyes and red legs."
start = time()
image = pipeline(prompt, num_inference_steps=inference_steps).images[0]
print(f'Compilation time is {time()-start} sec')
image.save("naruto.png")

start = time()
image = pipeline(prompt, num_inference_steps=inference_steps).images[0]
print(f'Inference time is {time()-start} sec after compilation')
```

</hfoption>

</hfoptions>

## الخطوات التالية

تهانينا على تدريب نموذج SDXL! لمعرفة المزيد عن كيفية استخدام نموذجك الجديد، قد تكون الأدلة التالية مفيدة:

- اقرأ دليل [Stable Diffusion XL](../using-diffusers/sdxl) لمعرفة كيفية استخدامه في مجموعة متنوعة من المهام المختلفة (النص إلى الصورة، والصورة إلى الصورة، والرسم)، وكيفية استخدام نموذج المُحسِّن الخاص به، وأنواع مختلفة من التكييفات الدقيقة.

- تحقق من أدلة التدريب [DreamBooth](dreambooth) و [LoRA](lora) لمعرفة كيفية تدريب نموذج SDXL مخصص باستخدام بضع صور فقط. يمكن حتى الجمع بين تقنيتي التدريب هاتين!