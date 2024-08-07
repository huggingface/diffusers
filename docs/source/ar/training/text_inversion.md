# الانعكاس النصي

[الانعكاس النصي](https://hf.co/papers/2208.01618) هي تقنية تدريب لشخصنة نماذج توليد الصور باستخدام عدد قليل فقط من صور الأمثلة لما تريد أن تتعلمه. تعمل هذه التقنية من خلال تعلم وتحديث تضمين النص (ترتبط التضمينات الجديدة بكلمة خاصة يجب استخدامها في الفكرة) لمطابقة صور الأمثلة التي توفرها.

إذا كنت تتدرب على معالج رسومات (GPU) ذي ذاكرة وصول عشوائي (VRAM) محدودة، فيجب عليك تجربة تمكين معلمات `gradient_checkpointing` و`mixed_precision` في أمر التدريب. يمكنك أيضًا تقليل البصمة الخاصة بك باستخدام الاهتمام الموفّر للذاكرة مع [xFormers](../optimization/xformers). يتم أيضًا دعم التدريب JAX/Flax للتدريب الفعال على وحدات معالجة الرسومات (TPUs) ووحدات معالجة الرسومات (GPUs)، ولكنه لا يدعم نقاط تفتيش التدرجات أو xFormers. باستخدام نفس التكوين والإعداد مثل PyTorch، يجب أن يكون نص Flax أسرع بنسبة 70% على الأقل!

سيتناول هذا الدليل النص البرمجي [textual_inversion.py](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py) لمساعدتك على التعرف عليه بشكل أفضل، وكيف يمكنك تكييفه مع حالتك الاستخدام.

قبل تشغيل النص البرمجي، تأكد من تثبيت المكتبة من المصدر:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

انتقل إلى مجلد المثال باستخدام نص التدريب وقم بتثبيت التبعيات المطلوبة للنص البرمجي الذي تستخدمه:

<hfoptions id="installation">
<hfoption id="PyTorch">

```bash
cd examples/textual_inversion
pip install -r requirements.txt
```

</hfoption>
<hfoption id="Flax">

```bash
cd examples/textual_inversion
pip install -r requirements_flax.txt
```

</hfoption>
</hfoptions>

🤗 Accelerate هي مكتبة للمساعدة في التدريب على وحدات معالجة الرسومات/وحدات معالجة الرسومات (GPUs/TPUs) متعددة أو باستخدام الدقة المختلطة. سيقوم تلقائيًا بتكوين إعداد التدريب الخاص بك بناءً على الأجهزة وبيئتك. الق نظرة على جولة 🤗 Accelerate [السريعة](https://huggingface.co/docs/accelerate/quicktour) لمعرفة المزيد.

قم بتهيئة بيئة 🤗 Accelerate:

```bash
accelerate config
```

لإعداد بيئة 🤗 Accelerate الافتراضية دون اختيار أي تكوينات:

```bash
accelerate config default
```

أو إذا لم يدعم بيئتك غلافًا تفاعليًا، مثل دفتر الملاحظات، فيمكنك استخدام:

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

أخيرًا، إذا كنت تريد تدريب نموذج على مجموعة البيانات الخاصة بك، فراجع دليل [إنشاء مجموعة بيانات للتدريب](create_dataset) لمعرفة كيفية إنشاء مجموعة بيانات تعمل مع نص التدريب.

تسلط الأقسام التالية الضوء على أجزاء من نص التدريب المهمة لفهم كيفية تعديلها، ولكنها لا تغطي كل جانب من جوانب النص البرمجي بالتفصيل. إذا كنت مهتمًا بمعرفة المزيد، فلا تتردد في قراءة النص البرمجي [هنا](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py) وأخبرنا إذا كان لديك أي أسئلة أو مخاوف.

## معلمات النص البرمجي

يحتوي نص التدريب على العديد من المعلمات لمساعدتك في تكييف عملية التدريب مع احتياجاتك. يتم سرد جميع المعلمات ووصفها في دالة [`parse_args()`](https://github.com/huggingface/diffusers/blob/839c2a5ece0af4e75530cb520d77bc7ed8acf474/examples/textual_inversion/textual_inversion.py#L176). حيثما ينطبق، توفر Diffusers القيم الافتراضية لكل معلمة، مثل حجم دفعة التدريب ومعدل التعلم، ولكن لا تتردد في تغيير هذه القيم في أمر التدريب إذا كنت ترغب في ذلك.

على سبيل المثال، لزيادة عدد خطوات تراكم التدرجات فوق القيمة الافتراضية 1:

```bash
accelerate launch textual_inversion.py \
--gradient_accumulation_steps=4
```

بعض المعلمات الأساسية والمهمة الأخرى التي يجب تحديدها:

- `--pretrained_model_name_or_path`: اسم النموذج على Hub أو مسار محلي للنموذج المدرب مسبقًا
- `--train_data_dir`: المسار إلى مجلد يحتوي على مجموعة بيانات التدريب (صور الأمثلة)
- `--output_dir`: المكان الذي سيتم فيه حفظ النموذج المدرب
- `--push_to_hub`: ما إذا كان سيتم دفع النموذج المدرب إلى Hub
- `--checkpointing_steps`: تكرار حفظ نقطة تفتيش أثناء تدريب النموذج؛ هذا مفيد إذا تم مقاطعة التدريب لأي سبب من الأسباب، فيمكنك الاستمرار في التدريب من تلك النقطة عن طريق إضافة `--resume_from_checkpoint` إلى أمر التدريب
- `--num_vectors`: عدد المتجهات لتعلم التضمينات بها؛ زيادة هذا المعلمة يساعد النموذج على التعلم بشكل أفضل ولكنه يأتي بتكاليف تدريب متزايدة
- `--placeholder_token`: الكلمة الخاصة لربط التضمينات المكتسبة (يجب استخدام الكلمة في فكرتك للاستدلال)
- `--initializer_token`: كلمة واحدة تصف بشكل عام الكائن أو الأسلوب الذي تحاول التدريب عليه
- `--learnable_property`: ما إذا كنت تدرب النموذج لتعلم "أسلوب" جديد (على سبيل المثال، أسلوب الرسم لفان جوخ) أو "كائن" (على سبيل المثال، كلبك)

## نص التدريب

على عكس بعض نصوص التدريب الأخرى، يستخدم نص textual_inversion.py فئة مجموعة بيانات مخصصة، [`TextualInversionDataset`](https://github.com/huggingface/diffusers/blob/b81c69e489aad3a0ba73798c459a33990dc4379c/examples/textual_inversion/textual_inversion.py#L487) لإنشاء مجموعة بيانات. يمكنك تخصيص حجم الصورة، والرمز النائب، وطريقة الاستيفاء، وما إذا كان سيتم اقتصاص الصورة، والمزيد. إذا كنت بحاجة إلى تغيير طريقة إنشاء مجموعة البيانات، فيمكنك تعديل `TextualInversionDataset`.

بعد ذلك، ستجد رمز معالجة مجموعة البيانات وحلقة التدريب في دالة [`main()`](https://github.com/huggingface/diffusers/blob/839c2a5ece0af4e75530cb520d77bc7ed8acf474/examples/textual_inversion/textual_inversion.py#L573).

يبدأ النص البرمجي بتحميل [الرموز](https://github.com/huggingface/diffusers/blob/b81c69e489aad3a0ba73798c459a33990dc4379c/examples/textual_inversion/textual_inversion.py#L616)، [المخطط والنماذج](https://github.com/huggingface/diffusers/blob/b81c69e489aad3a0ba73798c459a33990dc4379c/examples/textual_inversion/textual_inversion.py#L622):

```py
# تحميل الرموز
if args.tokenizer_name:
tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
elif args.pretrained_model_name_or_path:
tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

# تحميل المخطط والنماذج
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained(
args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
)
vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
unet = UNet2DConditionModel.from_pretrained(
args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
)
```

يتم إضافة الرمز النائب الخاص [هنا](https://github.com/huggingface/diffusers/blob/b81c69e489aad3a0ba73798c459a33990dc4379c/examples/textual_inversion/textual_inversion.py#L632) إلى الرموز، ويتم إعادة ضبط التضمين لمراعاة الرمز الجديد.

بعد ذلك، يقوم النص البرمجي [بإنشاء مجموعة بيانات](https://github.com/huggingface/diffusers/blob/b81c69e489aad3a0ba73798c459a33990dc4379c/examples/textual_inversion/textual_inversion.py#L716) من `TextualInversionDataset`:

```py
train_dataset = TextualInversionDataset(
data_root=args.train_data_dir,
tokenizer=tokenizer,
size=args.resolution,
placeholder_token=(" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids))),
repeats=args.repeats,
learnable_property=args.learnable_property,
center_crop=args.center_crop,
set="train",
)
train_dataloader = torch.utils.data.DataLoader(
train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
)
```

أخيرًا، تتولى حلقة [التدريب](https://github.com/huggingface/diffusers/blob/b81c69e489aad3a0ba73798c459a33990dc4379c/examples/textual_inversion/textual_inversion.py#L784) كل شيء آخر بدءًا من التنبؤ بقايا الضوضاء وحتى تحديث أوزان التضمين للرمز النائب الخاص.

إذا كنت تريد معرفة المزيد حول كيفية عمل حلقة التدريب، فراجع البرنامج التعليمي [Understanding pipelines, models and schedulers](../using-diffusers/write_own_pipeline) الذي يكسر النمط الأساسي لعملية إزالة الضوضاء.
## تشغيل السكربت

عندما تنتهي من إجراء جميع التغييرات أو تكون راضيًا عن التكوين الافتراضي، ستكون جاهزًا لتشغيل سكربت التدريب! 🚀

لأغراض هذا الدليل، ستقوم بتنزيل بعض الصور لـ [لعبة قط](https://huggingface.co/datasets/diffusers/cat_toy_example) وحفظها في دليل. ولكن تذكر أنه يمكنك إنشاء مجموعة بياناتك واستخدامها إذا أردت (راجع دليل [إنشاء مجموعة بيانات للتدريب](create_dataset)).

```py
from huggingface_hub import snapshot_download

local_dir = "./cat"
snapshot_download(
"diffusers/cat_toy_example", local_dir=local_dir, repo_type="dataset", ignore_patterns=".gitattributes"
)
```

قم بتعيين متغير البيئة `MODEL_NAME` إلى معرف نموذج على Hub أو مسار إلى نموذج محلي، و`DATA_DIR` إلى المسار الذي قمت بتنزيل صور القط إليه للتو. يقوم السكربت بإنشاء وحفظ الملفات التالية في مستودعك:

- `learned_embeds.bin`: متجهات التضمين المكتسبة المقابلة لصور المثال لديك
- `token_identifier.txt`: رمز المسافة الاحتياطية الخاص
- `type_of_concept.txt`: نوع المفهوم الذي تقوم بالتدريب عليه (إما "object" أو "style")

<Tip warning={true}>
تستغرق عملية التدريب الكاملة ~1 ساعة على GPU V100 واحد.
</Tip>

هناك شيء واحد قبل إطلاق السكربت. إذا كنت مهتمًا بمتابعة عملية التدريب، فيمكنك حفظ الصور المولدة بشكل دوري أثناء تقدم التدريب. أضف المعلمات التالية إلى أمر التدريب:

```bash
--validation_prompt="A <cat-toy> train"
--num_validation_images=4
--validation_steps=100
```

<hfoptions id="training-inference">
<hfoption id="PyTorch">

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./cat"

accelerate launch textual_inversion.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--train_data_dir=$DATA_DIR \
--learnable_property="object" \
--placeholder_token="<cat-toy>" \
--initializer_token="toy" \
--resolution=512 \
--train_batch_size=1 \
--gradient_accumulation_steps=4 \
--max_train_steps=3000 \
--learning_rate=5.0e-04 \
--scale_lr \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--output_dir="textual_inversion_cat" \
--push_to_hub
```

</hfoption>
<hfoption id="Flax">

```bash
export MODEL_partum="duongna/stable-diffusion-v1-4-flax"
export DATA_DIR="./cat"

python textual_inversion_flax.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--train_data_dir=$DATA_DIR \
--learnable_property="object" \
--placeholder_token="<cat-toy>" \
--initializer_token="toy" \
--resolution=512 \
--train_batch_size=1 \
--max_train_steps=3000 \
--learning_rate=5.0e-04 \
--scale_lr \
--output_dir="textual_inversion_cat" \
--push_to_hub
```

</hfoption>
</hfoptions>

بعد اكتمال التدريب، يمكنك استخدام نموذجك المدرب حديثًا للاستنتاج مثل:

<hfoptions id="training-inference">
<hfoption id="PyTorch">

```py
from diffusers import StableDiffusionPipeline
import torch

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipeline.load_textual_inversion("sd-concepts-library/cat-toy")
image = pipeline("A <cat-toy> train", num_inference_steps=50).images[0]
image.save("cat-train.png")
```

</hfoption>
<hfoption id="Flax">

لا يدعم Flax طريقة [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`]، ولكن يقوم سكربت textual_inversion_flax.py [بحفظ](https://github.com/huggingface/diffusers/blob/c0f058265161178f2a88849e92b37ffdc81f1dcc/examples/textual_inversion/textual_inversion_flax.py#L636C2-L636C2) التضمينات المكتسبة كجزء من النموذج بعد التدريب. وهذا يعني أنه يمكنك استخدام النموذج للاستنتاج مثل أي نموذج Flax آخر:

```py
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import FlaxStableDiffusionPipeline

model_path = "path-to-your-trained-model"
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(model_path, dtype=jax.numpy.bfloat16)

prompt = "A <cat-toy> train"
prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 50

num_samples = jax.device_count()
prompt = num_samples * [prompt]
prompt_ids = pipeline.prepare_inputs(prompt)

# تقسيم المدخلات و rng
params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
image.save("cat-train.png")
```

</hfoption>
</hfoptions>

## الخطوات التالية

تهانينا على تدريب نموذجك الخاص للانعكاس النصي! 🎉 لمعرفة المزيد حول كيفية استخدام نموذجك الجديد، قد تكون الأدلة التالية مفيدة:

- تعرف على كيفية [تحميل تضمينات الانعكاس النصي](../using-diffusers/loading_adapters) واستخدامها أيضًا كتضمينات سلبية.
- تعلم كيفية استخدام [الانعكاس النصي](textual_inversion_inference) للاستنتاج مع Stable Diffusion 1/2 و Stable Diffusion XL.