# DreamBooth

[DreamBooth](https://huggingface.co/papers/2208.12242) هي تقنية تدريب تقوم بتحديث نموذج الانتشار بالكامل من خلال التدريب على عدد قليل فقط من الصور لموضوع أو أسلوب معين. تعمل هذه التقنية من خلال ربط كلمة خاصة في النص الفوري بالصور المثالية.

إذا كنت تقوم بالتدريب على وحدة معالجة رسومات (GPU) ذات ذاكرة وصول عشوائي (VRAM) محدودة، فيجب عليك تجربة تمكين المعلمات `gradient_checkpointing` و`mixed_precision` في أمر التدريب. يمكنك أيضًا تقليل البصمة الذاكرية الخاصة بك باستخدام الانتباه الفعال للذاكرة مع [xFormers](../optimization/xformers). يتم أيضًا دعم التدريب JAX/Flax للتدريب الفعال على وحدات معالجة الرسومات (TPUs) ووحدات معالجة الرسومات (GPUs)، ولكنه لا يدعم نقاط تفتيش التدرج أو xFormers. يجب أن يكون لديك وحدة معالجة رسومات (GPU) بها أكثر من 30 جيجابايت من الذاكرة إذا كنت تريد التدريب بشكل أسرع باستخدام Flax.

سيتعمق هذا الدليل في دراسة النص البرمجي [train_dreambooth.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py) لمساعدتك على التعرف عليه بشكل أفضل، وكيف يمكنك تكييفه مع حالتك الاستخدامية الخاصة.

قبل تشغيل النص البرمجي، تأكد من تثبيت المكتبة من المصدر:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

انتقل إلى المجلد "مثال" بالنص البرمجي للتدريب وقم بتثبيت التبعيات المطلوبة للنص البرمجي الذي تستخدمه:

<hfoptions id="installation">
<hfoption id="PyTorch">

```bash
cd examples/dreambooth
pip install -r requirements.txt
```

</hfoption>
<hfoption id="Flax">

```bash
cd examples/dreambooth
pip install -r requirements_flax.txt
```

</hfoption>
</hfoptions>

<Tip>

🤗 Accelerate هي مكتبة تساعدك على التدريب على وحدات معالجة رسومات (GPU) / وحدات معالجة الرسومات (TPUs) متعددة أو باستخدام الدقة المختلطة. سيقوم تلقائيًا بتكوين إعداد التدريب الخاص بك بناءً على الأجهزة وبيئة العمل لديك. اطلع على الجولة السريعة لـ 🤗 Accelerate [Quick tour](https://huggingface.co/docs/accelerate/quicktour) لمعرفة المزيد.

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

أخيرًا، إذا كنت تريد تدريب نموذج على مجموعة البيانات الخاصة بك، فراجع دليل [إنشاء مجموعة بيانات للتدريب](create_dataset) لمعرفة كيفية إنشاء مجموعة بيانات تعمل مع نص التدريب البرمجي.

<Tip>

تسلط الأقسام التالية الضوء على أجزاء من نص التدريب البرمجي المهمة لفهم كيفية تعديلها، ولكنها لا تغطي كل جانب من جوانب النص البرمجي بالتفصيل. إذا كنت مهتمًا بمعرفة المزيد، فلا تتردد في قراءة النص البرمجي [script](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py) وأخبرنا إذا كان لديك أي أسئلة أو مخاوف.

</Tip>

## معلمات النص البرمجي

<Tip warning={true}>

DreamBooth حساس للغاية لمعلمات التدريب، ومن السهل أن يحدث بها زيادة في التكيّف. اقرأ منشور المدونة [تدريب Stable Diffusion باستخدام Dreambooth باستخدام 🧨 Diffusers](https://huggingface.co/blog/dreambooth) لمعرفة الإعدادات الموصى بها لمواضيع مختلفة لمساعدتك في اختيار المعلمات المناسبة.

</Tip>

يوفر نص التدريب البرمجي العديد من المعلمات لتخصيص عملية التدريب. يمكن العثور على جميع المعلمات ووصفها في دالة [`parse_args()`](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L228). يتم تعيين المعلمات بقيم افتراضية يجب أن تعمل بشكل جيد دون أي تكوين، ولكن يمكنك أيضًا تعيين قيمك الخاصة في أمر التدريب إذا كنت ترغب في ذلك.

على سبيل المثال، للتدريب بتنسيق bf16:

```bash
accelerate launch train_dreambooth.py \
--mixed_precision="bf16"
```

بعض المعلمات الأساسية والمهمة التي يجب معرفتها وتحديدها هي:

- `--pretrained_model_name_or_path`: اسم النموذج على Hub أو مسار محلي للنموذج الذي تم تدريبه مسبقًا
- `--instance_data_dir`: المسار إلى المجلد الذي يحتوي على مجموعة بيانات التدريب (صور المثال)
- `--instance_prompt`: النص الفوري الذي يحتوي على الكلمة الخاصة لصور المثال
- `--train_text_encoder`: ما إذا كان سيتم أيضًا تدريب مشفر النص
- `--output_dir`: المكان الذي سيتم فيه حفظ النموذج الذي تم تدريبه
- `--push_to_hub`: ما إذا كان سيتم دفع النموذج الذي تم تدريبه إلى Hub
- `--checkpointing_steps`: تكرار حفظ نقطة تفتيش أثناء تدريب النموذج؛ هذا مفيد إذا تم مقاطعة التدريب لسبب ما، فيمكنك الاستمرار في التدريب من تلك النقطة عن طريق إضافة `--resume_from_checkpoint` إلى أمر التدريب

### وزن Min-SNR

يمكن أن تساعد استراتيجية الوزن Min-SNR في التدريب عن طريق إعادة توازن الخسارة لتحقيق تقارب أسرع. يدعم نص التدريب البرمجي التنبؤ بـ `epsilon` (الضوضاء) أو `v_prediction`، ولكن Min-SNR متوافق مع كلا نوعي التنبؤ. استراتيجية الوزن هذه مدعومة فقط بواسطة PyTorch وغير متوفرة في نص التدريب البرمجي Flax.

أضف المعلمة `--snr_gamma` وقم بتعيينها على القيمة الموصى بها 5.0:

```bash
accelerate launch train_dreambooth.py \
--snr_gamma=5.0
```

### خسارة الحفاظ على الأولوية

خسارة الحفاظ على الأولوية هي طريقة تستخدم عينات مولدة من النموذج نفسه لمساعدته على تعلم كيفية إنشاء صور أكثر تنوعًا. نظرًا لأن صور العينات المولدة هذه تنتمي إلى نفس الفئة التي تنتمي إليها الصور التي قدمتها، فإنها تساعد النموذج على الاحتفاظ بما تعلمه عن الفئة وكيف يمكنه استخدام ما يعرفه بالفعل عن الفئة لإنشاء تكوينات جديدة.

- `--with_prior_preservation`: ما إذا كان سيتم استخدام خسارة الحفاظ على الأولوية
- `--prior_loss_weight`: يتحكم في تأثير خسارة الحفاظ على الأولوية على النموذج
- `--class_data_dir`: المسار إلى المجلد الذي يحتوي على صور العينات المولدة للفئة
- `--class_prompt`: النص الفوري الذي يصف فئة صور العينات المولدة

```bash
accelerate launch train_dreambooth.py \
--with_prior_preservation \
--prior_loss_weight=1.0 \
--class_data_dir="path/to/class/images" \
--class_prompt="text prompt describing class"
```

### تدريب مشفر النص

لتحسين جودة المخرجات المولدة، يمكنك أيضًا تدريب مشفر النص بالإضافة إلى UNet. يتطلب ذلك ذاكرة إضافية وتحتاج إلى وحدة معالجة رسومات (GPU) بها 24 جيجابايت على الأقل من ذاكرة الوصول العشوائي (VRAM). إذا كان لديك الأجهزة اللازمة، فإن تدريب مشفر النص ينتج نتائج أفضل، خاصة عند إنشاء صور الوجوه. قم بتمكين هذا الخيار عن طريق:

```bash
accelerate launch train_dreambooth.py \
--train_text_encoder
```

## نص التدريب البرمجي

يأتي DreamBooth بمجموعات البيانات الخاصة به:

- [`DreamBoothDataset`](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L604): معالجة الصور المسبقة وصور الفئة، ورموز النص الفوري للتدريب
- [`PromptDataset`](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L738): إنشاء تضمين النص الفوري لإنشاء صور الفئة

إذا قمت بتمكين [خسارة الحفاظ على الأولوية](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L842)، يتم إنشاء صور الفئة هنا:

```py
sample_dataset = PromptDataset(args.class_prompt, num_new_images)
sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

sample_dataloader = accelerator.prepare(sample_dataloader)
pipeline.to(accelerator.device)

for example in tqdm(
sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
):
images = pipeline(example["prompt"]).images
```

بعد ذلك، تأتي دالة [`main()`](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L799) التي تتولى إعداد مجموعة البيانات للتدريب وحلقة التدريب نفسها. يقوم النص البرمجي بتحميل [المحول البرمجي](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L898)، [المخطط الزمني](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L912C1-L912C1) والنماذج:

```py
# تحميل المحول البرمجي
if args.tokenizer_name:
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
elif args.pretrained_model_name_or_path:
tokenizer = AutoTokenizer.from_pretrained(
args.pretrained_model_name_or_path,
subfolder="tokenizer",
revision=args.revision,
use_fast=False,
)

# تحميل المخطط الزمني والنماذج
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
text_encoder = text_encoder_cls.from_pretrained(
args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
)

if model_has_vae(args):
vae = AutoencoderKL.from_pretrained(
args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
)
else:
vae = None

unet = UNet2DConditionModel.from_pretrained(
args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
)
```

بعد ذلك، حان الوقت [لإنشاء مجموعة بيانات التدريب](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L1073) وDataLoader من `DreamBoothDataset`:

```py
train_dataset = DreamBoothDataset(
instance_data_root=args.instance_data_dir,
instance_prompt=args.instance_prompt,
class_data_root=args.class_data_dir if args.with_prior_preservation else None,
class_prompt=args.class_prompt,
class_num=args.num_class_images,
tokenizer=tokenizer,
size=args.resolution,
center_crop=args.center_crop,
encoder_hidden_states=pre_computed_encoder_hidden_states,
class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
tokenizer_max_length=args.tokenizer_max_length,
)

train_dataloader = torch.utils.data.DataLoader(
train_dataset,
batch_size=args.train_batch_size,
shuffle=True,
collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
num_workers=args.dataloader_num_workers,
)
```

أخيرًا، تتولى حلقة التدريب [الخطوات](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L1151) المتبقية مثل تحويل الصور إلى مساحة خفية، وإضافة ضوضاء إلى الإدخال، والتنبؤ ببقايا الضوضاء، وحساب الخسارة.

إذا كنت تريد معرفة المزيد حول كيفية عمل حلقة التدريب، فراجع البرنامج التعليمي [فهم خطوط الأنابيب والنماذج والمخططات الزمنية](../using-diffusers/write_own_pipeline) الذي يكسر النمط الأساسي لعملية إزالة الضوضاء.
## تشغيل السكربت 

الآن أنت مستعد لتشغيل سكربت التدريب! 🚀 

في هذا الدليل، ستقوم بتنزيل بعض الصور لكلب [dog](https://huggingface.co/datasets/diffusers/dog-example) وحفظها في دليل. ولكن تذكر، يمكنك إنشاء واستخدام مجموعة بياناتك الخاصة إذا أردت (راجع الدليل [إنشاء مجموعة بيانات للتدريب](create_dataset)).

```py
from huggingface_hub import snapshot_download

local_dir = "./dog"
snapshot_download(
"diffusers/dog-example",
local_dir=local_dir,
repo_type="dataset",
ignore_patterns=".gitattributes",
)
```

قم بضبط متغير البيئة `MODEL_NAME` إلى معرف نموذج على هب أو مسار إلى نموذج محلي، و`INSTANCE_DIR` إلى المسار الذي قمت بتنزيل صور الكلب إليه، و`OUTPUT_DIR` إلى المكان الذي تريد حفظ النموذج فيه. ستستخدم "sks" ككلمة خاصة لربط التدريب بها.

إذا كنت مهتمًا بمتابعة عملية التدريب، فيمكنك حفظ الصور المولدة بشكل دوري أثناء تقدم التدريب. أضف المعلمات التالية إلى أمر التدريب:

```bash
--validation_prompt="a photo of a sks dog"
--num_validation_images=4
--validation_steps=100
```

هناك شيء واحد قبل تشغيل السكربت! اعتمادًا على وحدة معالجة الرسوميات (GPU) التي لديك، قد تحتاج إلى تمكين بعض التحسينات لتدريب DreamBooth.

<hfoptions id="gpu-select">
<hfoption id="16GB">

على وحدة معالجة الرسوميات (GPU) بسعة 16 جيجابايت، يمكنك استخدام محسن bitsandbytes 8-bit ومحسن نقاط التفتيش التدريجي للمساعدة في تدريب نموذج DreamBooth. قم بتثبيت bitsandbytes:

```py
pip install bitsandbytes
```

بعد ذلك، أضف المعلمة التالية إلى أمر التدريب:

```bash
accelerate launch train_dreambooth.py \
--gradient_checkpointing \
--use_8bit_adam \
```

</hfoption>
<hfoption id="12GB">

على وحدة معالجة الرسوميات (GPU) بسعة 12 جيجابايت، ستحتاج إلى محسن bitsandbytes 8-bit ومحسن نقاط التفتيش التدريجي، وxFormers، وضبط المتدرجات على `None` بدلاً من الصفر لتقليل استخدام الذاكرة.

```bash
accelerate launch train_dreambooth.py \
--use_8bit_adam \
--gradient_checkpointing \
--enable_xformers_memory_efficient_attention \
--set_grads_to_none \
```

</hfoption>
<hfoption id="8GB">

على وحدة معالجة الرسوميات (GPU) بسعة 8 جيجابايت، ستحتاج إلى [DeepSpeed](https://www.deepspeed.ai/) لنقل بعض المصفوفات من ذاكرة الوصول العشوائي للرسوميات (VRAM) إلى وحدة المعالجة المركزية (CPU) أو NVME للسماح بالتدريب باستخدام ذاكرة GPU أقل.

قم بتشغيل الأمر التالي لتكوين بيئة 🤗 Accelerate:

```bash
accelerate config
```

أثناء التكوين، تأكد من أنك تريد استخدام DeepSpeed. الآن يجب أن يكون من الممكن التدريب على أقل من 8 جيجابايت من ذاكرة الوصول العشوائي للرسوميات (VRAM) من خلال الجمع بين DeepSpeed stage 2، والدقة المختلطة fp16، ونقل معلمات النموذج وحالة المحسن إلى وحدة المعالجة المركزية (CPU). تتمثل السلبية في أن هذا يتطلب المزيد من ذاكرة الوصول العشوائي (RAM) للنظام (~25 جيجابايت). راجع وثائق DeepSpeed للحصول على خيارات تكوين إضافية.

يجب عليك أيضًا تغيير محسن Adam الافتراضي إلى الإصدار الأمثل لـ DeepSpeed من Adam [`deepspeed.ops.adam.DeepSpeedCPUAdam`](https://deepspeed.readthedocs.io/en/latest/optimizers.html#adam-cpu) للحصول على تسريع كبير. يتطلب تمكين `DeepSpeedCPUAdam` أن يكون إصدار مجموعة أدوات CUDA في نظامك هو نفسه المثبت مع PyTorch.

لا يبدو أن محسنات bitsandbytes 8-bit متوافقة مع DeepSpeed في الوقت الحالي.

هذا كل شيء! لا تحتاج إلى إضافة أي معلمات إضافية إلى أمر التدريب الخاص بك.

</hfoption>
</hfoptions>

<hfoptions id="training-inference">
<hfoption id="PyTorch">

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./dog"
export OUTPUT_DIR="path_to_saved_model"

accelerate launch train_dreambooth.py \
--pretrained_model_name_or_path=$MODEL_NAME  \
--instance_data_dir=$INSTANCE_DIR \
--output_dir=$OUTPUT_DIR \
--instance_prompt="a photo of sks dog" \
--resolution=512 \
--train_batch_size=1 \
--gradient_accumulation_steps=1 \
--learning_rate=5e-6 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--max_train_steps=400 \
--push_to_hub
```

</hfoption>
<hfoption id="Flax">

```bash
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_DIR="./dog"
export OUTPUT_DIR="path-to-save-model"

python train_dreambooth_flax.py \
--pretrained_model_name_or_path=$MODEL_NAME  \
--instance_data_dir=$INSTANCE_DIR \
--output_dir=$OUTPUT_DIR \
--instance_prompt="a photo of sks dog" \
--resolution=512 \
--train_batch_size=1 \
--learning_rate=5e-6 \
--max_train_steps=400 \
--push_to_hub
```

</hfoption>
</hfoptions>

بمجرد اكتمال التدريب، يمكنك استخدام نموذجك المدرب حديثًا للاستنتاج!

<Tip>

هل لا يمكنك الانتظار لتجربة نموذجك للاستنتاج قبل اكتمال التدريب؟ 🤭 تأكد من أن لديك أحدث إصدار من 🤗 Accelerate مثبتًا.

```py
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch

unet = UNet2DConditionModel.from_pretrained("path/to/model/checkpoint-100/unet")

# if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
text_encoder = CLIPTextModel.from_pretrained("path/to/model/checkpoint-100/checkpoint-100/text_encoder")

pipeline = DiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5", unet=unet, text_encoder=text_encoder, dtype=torch.float16,
).to("cuda")

image = pipeline("A photo of sks dog in a bucket", num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("dog-bucket.png")
```

</Tip>

<hfoptions id="training-inference">
<hfoption id="PyTorch">

```py
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("path_to_saved_model", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
image = pipeline("A photo of sks dog in a bucket", num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("dog-bucket.png")
```

</hfoption>
<hfoption id="Flax">

```py
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import FlaxStableDiffusionPipeline

pipeline, params = FlaxStableDiffusionPipeline.from_pretrained("path-to-your-trained-model", dtype=jax.numpy.bfloat16)

prompt = "A photo of sks dog in a bucket"
prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 50

num_samples = jax.device_count()
prompt = num_samples * [prompt]
prompt_ids = pipeline.prepare_inputs(prompt)

# shard inputs and rng
params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
image.save("dog-bucket.png")
```

</hfoption>
</hfoptions>

## لورا

لورا (LoRA) هي تقنية تدريب لتخفيض عدد المعلمات القابلة للتدريب بشكل كبير. ونتيجة لذلك، يكون التدريب أسرع ويكون من الأسهل تخزين الأوزان الناتجة لأنها أصغر بكثير (~100 ميجابايت). استخدم السكربت [train_dreambooth_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py) لتدريب لورا.

يناقش سكربت التدريب لورا بالتفصيل في دليل [تدريب لورا](lora).

## ستبل ديفيشن إكس إل

ستبل ديفيشن إكس إل (SDXL) هو نموذج قوي للصور النصية ينشئ صورًا عالية الدقة، ويضيف مشفر نص ثانٍ إلى بنائه. استخدم السكربت [train_dreambooth_lora_sdxl.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_sdxl.py) لتدريب نموذج SDXL باستخدام لورا.

يناقش سكربت التدريب SDXL بالتفصيل في دليل [تدريب SDXL](sdxl).
## DeepFloyd IF

نموذج DeepFloyd IF هو نموذج تسريب بكسل متتالي بثلاث مراحل. تقوم المرحلة الأولى بتوليد صورة أساسية، بينما تقوم المرحلتان الثانية والثالثة بتكبير حجم الصورة الأساسية تدريجياً إلى صورة عالية الدقة بحجم 1024x1024 بكسل. استخدم النصوص البرمجية [train_dreambooth_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py) أو [train_dreambooth.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py) لتدريب نموذج DeepFloyd IF باستخدام LoRA أو النموذج الكامل.

يستخدم DeepFloyd IF التباين المتوقع، ولكن النصوص البرمجية لتدريب Diffusers تستخدم الخطأ المتوقع، لذلك يتم تحويل النماذج المدربة من DeepFloyd IF إلى جدول تباين ثابت. ستقوم النصوص البرمجية للتدريب بتحديث تكوين جدول مواعيد النموذج المدرب بالكامل نيابة عنك. ومع ذلك، عند تحميل أوزان LoRA المحفوظة، يجب أيضًا تحديث تكوين جدول مواعيد الأنبوب.

```py
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", use_safetensors=True)

pipe.load_lora_weights("<lora weights path>")

# تحديث جدول مواعيد التكوين إلى جدول تباين ثابت
pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config, variance_type="fixed_small")
```

يتطلب نموذج المرحلة 2 صور تحقق إضافية للتكبير. يمكنك تنزيل واستخدام نسخة مصغرة من صور التدريب لهذا الغرض.

```py
from huggingface_hub import snapshot_download

local_dir = "./dog_downsized"
snapshot_download(
"diffusers/dog-example-downsized",
local_dir=local_dir,
repo_type="dataset",
ignore_patterns=".gitattributes",
)
```

تقدم عينات الشفرة أدناه نظرة عامة موجزة حول كيفية تدريب نموذج DeepFloyd IF باستخدام مزيج من DreamBooth وLoRA. بعض المعلمات المهمة التي يجب ملاحظتها هي:

* `--resolution=64`، مطلوب دقة أصغر بكثير لأن DeepFloyd IF هو نموذج تسريب بكسل، وللعمل على البكسلات غير المضغوطة، يجب أن تكون صور الإدخال أصغر.
* `--pre_compute_text_embeddings`، احسب تضمين النص مسبقًا لتوفير الذاكرة لأن [`~transformers.T5Model`] يمكن أن يستهلك الكثير من الذاكرة.
* `--tokenizer_max_length=77`، يمكنك استخدام طول نص افتراضي أطول مع T5 كمحول نصي، ولكن إجراء الترميز الافتراضي للنموذج يستخدم طول نص أقصر.
* `--text_encoder_use_attention_mask`، لإرسال قناع الاهتمام إلى المحول النصي.

<hfoptions id="IF-DreamBooth">
<hfoption id="Stage 1 LoRA DreamBooth">

يتطلب تدريب المرحلة 1 من DeepFloyd IF باستخدام LoRA وDreamBooth حوالي 28 جيجابايت من الذاكرة.

```bash
export MODEL_NAME="DeepFloyd/IF-I-XL-v1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth_dog_lora"

accelerate launch train_dreambooth_lora.py \
--report_to wandb \
--pretrained_model_name_or_path=$MODEL_NAME \
--instance_data_dir=$INSTANCE_DIR \
--output_dir=$OUTPUT_DIR \
--instance_prompt="a sks dog" \
--resolution=64 \
--train_batch_size=4 \
--gradient_accumulation_steps=1 \
--learning_rate=5e-6 \
--scale_lr \
--max_train_steps=1200 \
--validation_prompt="a sks dog" \
--validation_epochs=25 \
--checkpointing_steps=100 \
--pre_compute_text_embeddings \
--tokenizer_max_length=77 \
--text_encoder_use_attention_mask
```

</hfoption>
<hfoption id="Stage 2 LoRA DreamBooth">

بالنسبة للمرحلة 2 من DeepFloyd IF مع LoRA وDreamBooth، انتبه إلى هذه المعلمات:

* `--validation_images`، الصور التي سيتم تكبيرها أثناء التحقق من الصحة.
* `--class_labels_conditioning=timesteps`، لشرط UNet الإضافي كما هو مطلوب في المرحلة 2.
* `--learning_rate=1e-6`، يتم استخدام معدل تعلم أقل مقارنة بالمرحلة 1.
* `--resolution=256`، الدقة المتوقعة لمكبر الحجم.

```bash
export MODEL_NAME="DeepFloyd/IF-II-L-v1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth_dog_upscale"
export VALIDATION_IMAGES="dog_downsized/image_1.png dog_downsized/image_2.png dog_downsized/image_3.png dog_downsized/image_4.png"

python train_dreambooth_lora.py \
--report_to wandb \
--pretrained_model_name_or_path=$MODEL_NAME \
--instance_data_dir=$INSTANCE_DIR \
--output_dir=$OUTPUT_DIR \
--instance_prompt="a sks dog" \
--resolution=256 \
--train_batch_size=4 \
--gradient_accumulation_steps=1 \
--learning_rate=1e-6 \
--max_train_steps=2000 \
--validation_prompt="a sks dog" \
--validation_epochs=100 \
--checkpointing_steps=500 \
--pre_compute_text_embeddings \
--tokenizer_max_length=77 \
--text_encoder_use_attention_mask \
--validation_images $VALIDATION_IMAGES \
--class_labels_conditioning=timesteps
```

</hfoption>
<hfoption id="Stage 1 DreamBooth">

بالنسبة للمرحلة 1 من DeepFloyd IF مع DreamBooth، انتبه إلى هذه المعلمات:

* `--skip_save_text_encoder`، لتخطي حفظ محول النص T5 الكامل مع النموذج الدقيق الضبط.
* `--use_8bit_adam`، لاستخدام محسن Adam ببت 8 لتوفير الذاكرة بسبب حجم حالة المحسن عند تدريب النموذج الكامل.
* `--learning_rate=1e-7`، يجب استخدام معدل تعلم منخفض للغاية لتدريب النموذج الكامل، وإلا ستتدهور جودة النموذج (يمكنك استخدام معدل تعلم أعلى مع حجم دفعة أكبر).

يمكن تدريب النموذج الكامل باستخدام محسن Adam ببت 8 وحجم دفعة يبلغ 4 باستخدام حوالي 48 جيجابايت من الذاكرة.

```bash
export MODEL_NAME="DeepFloyd/IF-I-XL-v1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth_if"

accelerate launch train_dreambooth.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--instance_data_dir=$INSTANCE_DIR \
--output_dir=$OUTPUT_DIR \
--instance_prompt="a photo of sks dog" \
--resolution=64 \
--train_batch_size=4 \
--gradient_accumulation_steps=1 \
--learning_rate=1e-7 \
--max_train_steps=150 \
--validation_prompt "a photo of sks dog" \
--validation_steps 25 \
--text_encoder_use_attention_mask \
--tokenizer_max_length 77 \
--pre_compute_text_embeddings \
--use_8bit_adam \
--set_grads_to_none \
--skip_save_text_encoder \
--push_to_hub
```

</hfoption>
<hfoption id="Stage 2 DreamBooth">

بالنسبة للمرحلة 2 من DeepFloyd IF مع DreamBooth، انتبه إلى هذه المعلمات:

* `--learning_rate=5e-6`، استخدم معدل تعلم أقل مع حجم دفعة فعال أصغر.
* `--resolution=256`، الدقة المتوقعة لمكبر الحجم.
* `--train_batch_size=2` و `--gradient_accumulation_steps=6`، لتدريب فعال على الصور التي تحتوي على وجوه، مطلوب أحجام دفعات أكبر.

```bash
export MODEL_NAME="DeepFloyd/IF-II-L-v1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth_dog_upscale"
export VALIDATION_IMAGES="dog_downsized/image_1.png dog_downsized/image_2.png dog_downsized/image_3.png dog_downsized/image_4.png"

accelerate launch train_dreambooth.py \
--report_to wandb \
--pretrained_model_name_or_path=$MODEL_NAME \
--instance_data_dir=$INSTANCE_DIR \
--output_dir=$OUTPUT_DIR \
--instance_prompt="a sks dog" \
--resolution=256 \
--train_batch_size=2 \
--gradient_accumulation_steps=6 \
--learning_rate=5e-6 \
--max_train_steps=2000 \
--validation_prompt="a sks dog" \
--validation_steps=150 \
--checkpointing_steps=500 \
--pre_compute_text_embeddings \
--tokenizer_max_length=77 \
--text_encoder_use_attention_mask \
--validation_images $VALIDATION_IMAGES \
--class_labels_conditioning timesteps \
--push_to_hub
```

</hfoption>
</hfoptions>

### نصائح التدريب

يمكن أن يكون تدريب نموذج DeepFloyd IF أمرًا صعبًا، ولكن فيما يلي بعض النصائح التي وجدناها مفيدة:

- LoRA كافٍ لتدريب نموذج المرحلة 1 لأن الدقة المنخفضة للنموذج تجعل من الصعب تمثيل التفاصيل الدقيقة على أي حال.
- بالنسبة للأشياء الشائعة أو البسيطة، لا تحتاج بالضرورة إلى ضبط مكبر الحجم بشكل دقيق. تأكد من تعديل المحث الذي يتم تمريره إلى مكبر الحجم لإزالة الرمز الجديد من المحث الخاص بالمرحلة 1. على سبيل المثال، إذا كان محث المرحلة 1 الخاص بك هو "a sks dog"، فيجب أن يكون محث المرحلة 2 الخاص بك "a dog".
- بالنسبة للتفاصيل الدقيقة مثل الوجوه، فإن ضبط مكبر الحجم للمرحلة 2 بشكل دقيق أفضل من تدريب نموذج المرحلة 2 باستخدام LoRA. كما يساعد استخدام معدلات تعلم أقل مع أحجام دفعات أكبر.
- يجب استخدام معدلات تعلم أقل لتدريب نموذج المرحلة 2.
- يعمل [`DDPMScheduler`] بشكل أفضل من DPMSolver المستخدم في النصوص البرمجية للتدريب.

## الخطوات التالية

تهانينا على تدريب نموذج DreamBooth الخاص بك! لمعرفة المزيد حول كيفية استخدام نموذجك الجديد، قد يكون الدليل التالي مفيدًا:

- تعرف على كيفية [تحميل نموذج DreamBooth](../using-diffusers/loading_adapters) للتنبؤ إذا كنت قد دربته باستخدام LoRA.