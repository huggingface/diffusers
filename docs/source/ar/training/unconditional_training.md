# إنشاء الصور غير المشروطة

لا تخضع نماذج إنشاء الصور غير المشروطة للظروف النصية أو الصورية أثناء التدريب. فهو يقوم فقط بتوليد صور تشبه توزيع بيانات التدريب الخاصة به.

سيتناول هذا الدليل برنامج النص البرمجي [train_unconditional.py](https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py) لمساعدتك على التعرف عليه، وكيف يمكنك تكييفه مع حالتك الاستخدام الخاصة بك.

قبل تشغيل النص البرمجي، تأكد من تثبيت المكتبة من المصدر:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

بعد ذلك، انتقل إلى مجلد المثال الذي يحتوي على نص التدريب البرمجي وقم بتثبيت التبعيات المطلوبة:

```bash
cd examples/unconditional_image_generation
pip install -r requirements.txt
```

🤗 Accelerate هي مكتبة تساعدك على التدريب على وحدات معالجة الرسوميات (GPU) أو وحدات معالجة الرسوميات ذات النطاق الترددي العالي (TPU) أو باستخدام الدقة المختلطة. سيقوم تلقائيًا بتكوين إعداد التدريب الخاص بك بناءً على الأجهزة وبيئتك. الق نظرة على الجولة السريعة لـ 🤗 Accelerate [Quick tour](https://huggingface.co/docs/accelerate/quicktour) لمعرفة المزيد.

قم بتهيئة بيئة 🤗 Accelerate:

```bash
accelerate config
```

لإعداد بيئة 🤗 Accelerate الافتراضية دون اختيار أي تكوينات:

```bash
accelerate config default
```

أو إذا لم يدعم بيئتك غلافًا تفاعليًا مثل دفتر الملاحظات، فيمكنك استخدام ما يلي:

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

أخيرًا، إذا كنت تريد تدريب نموذج على مجموعة البيانات الخاصة بك، فراجع دليل [إنشاء مجموعة بيانات للتدريب](create_dataset) لمعرفة كيفية إنشاء مجموعة بيانات تعمل مع نص التدريب البرمجي.

## معلمات النص البرمجي

تسلط الأقسام التالية الضوء على أجزاء من نص التدريب البرمجي المهمة لفهم كيفية تعديلها، ولكنها لا تغطي كل جانب من جوانب النص البرمجي بالتفصيل. إذا كنت مهتمًا بمعرفة المزيد، فلا تتردد في قراءة النص البرمجي [script](https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py) وأخبرنا إذا كان لديك أي أسئلة أو مخاوف.

يوفر نص التدريب البرمجي العديد من المعلمات لمساعدتك على تخصيص عملية تشغيل التدريب. يمكن العثور على جميع المعلمات ووصفاتها في دالة [`parse_args()`](https://github.com/huggingface/diffusers/blob/096f84b05f9514fae9f185cbec0a4d38fbad9919/examples/unconditional_image_generation/train_unconditional.py#L55). يوفر قيم افتراضية لكل معلمة، مثل حجم دفعة التدريب ومعدل التعلم، ولكن يمكنك أيضًا تعيين قيمك الخاصة في أمر التدريب إذا رغبت في ذلك.

على سبيل المثال، لإجراء التدريب باستخدام الدقة المختلطة بتنسيق bf16، أضف المعلمة `--mixed_precision` إلى أمر التدريب:

```bash
accelerate launch train_unconditional.py \
--mixed_precision="bf16"
```

تتضمن بعض المعلمات الأساسية والمهمة ما يلي:

- `--dataset_name`: اسم مجموعة البيانات على Hub أو مسار محلي إلى مجموعة البيانات التي سيتم التدريب عليها
- `--output_dir`: المكان الذي سيتم فيه حفظ النموذج المدرب
- `--push_to_hub`: ما إذا كان سيتم دفع النموذج المدرب إلى Hub
- `--checkpointing_steps`: تكرار حفظ نقطة تفتيش أثناء تدريب النموذج؛ هذا مفيد إذا تم مقاطعة التدريب، فيمكنك الاستمرار في التدريب من تلك النقطة عن طريق إضافة `--resume_from_checkpoint` إلى أمر التدريب

أحضر مجموعة البيانات الخاصة بك، ودع نص التدريب البرمجي يتعامل مع كل شيء آخر!

## نص التدريب البرمجي

يمكن العثور على رمز معالجة مجموعة البيانات وحلقة التدريب في دالة [`main()`](https://github.com/huggingface/diffusers/blob/096f84b05f9514fae9f84cbec0a4d38fbad9919/examples/unconditional_image_generation/train_unconditional.py#L275). إذا كنت بحاجة إلى تكييف نص التدريب البرمجي، فهذا هو المكان الذي ستحتاج إلى إجراء تغييراتك فيه.

ينشئ نص البرنامج النصي `train_unconditional` [نموذج `UNet2DModel`](https://github.com/huggingface/diffusers/blob/096f84b05f9514fae9f84cbec0a4d38fbad9919/examples/unconditional_image_generation/train_unconditional.py#L356) إذا لم توفر تكوين نموذج. يمكنك تكوين UNet هنا إذا أردت:

```py
model = UNet2DModel(
sample_size=args.resolution,
in_channels=3,
out_channels=3,
layers_per_block=2,
block_out_channels=(128, 128, 256, 256, 512, 512),
down_block_types=(
"DownBlock2D",
"DownBlock2D",
"DownBlock2D",
"DownBlock2D",
"AttnDownBlock2D",
"DownBlock2D",
),
up_block_types=(
"UpBlock2D",
"AttnUpBlock2D",
"UpBlock2D",
"UpBlock2D",
"UpBlock2D",
"UpBlock2D",
),
)
```

بعد ذلك، يقوم النص البرمجي بتهيئة [جدول](https://github.com/huggingface/diffusers/blob/096f84b05f9514fae9f84cbec0a4d38fbad9919/examples/unconditional_image_generation/train_unconditional.py#L418) و [محسن](https://github.com/huggingface/diffusers/blob/096f84b05f9514fae9f84cbec0a4d38fbad9919/examples/unconditional_image_generation/train_unconditional.py#L429):

```py
# Initialize the scheduler
accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
if accepts_prediction_type:
noise_scheduler = DDPMScheduler(
num_train_timesteps=args.ddpm_num_steps,
beta_schedule=args.ddpm_beta_schedule,
prediction_type=args.prediction_type,
)
else:
noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)

# Initialize the optimizer
optimizer = torch.optim.AdamW(
model.parameters(),
lr=args.learning_rate,
betas=(args.adam_beta1, args.adam_beta2),
weight_decay=args.adam_weight_decay,
eps=args.adam_epsilon,
)
```

ثم يقوم بتحميل [مجموعة بيانات](https://github.com/huggingface/diffusers/blob/096f84b05f9514fae9f84cbec0a4d38fbad9919/examples/unconditional_image_generation/train_unconditional.py#L451) ويمكنك تحديد كيفية [معالجتها مسبقًا](https://github.com/huggingface/diffusers/blob/096f84b05f9514fae9f84cbec0a4d38fbad9919/examples/unconditional_image_generation/train_unconditional.py#L455):

```py
dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")

augmentations = transforms.Compose(
[
transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
transforms.ToTensor(),
transforms.Normalize([0.5], [0.5]),
]
)
```

أخيرًا، تتولى [حلقة التدريب](https://github.com/huggingface/diffusers/blob/096f84b05f9514fae9f84cbec0a4d38fbad9919/examples/unconditional_image_generation/train_unconditional.py#L540) كل شيء آخر مثل إضافة الضوضاء إلى الصور، والتنبؤ ببقايا الضوضاء، وحساب الخسارة، وحفظ نقاط التفتيش في الخطوات المحددة، وحفظ النموذج ودفعه إلى Hub. إذا كنت تريد معرفة المزيد حول كيفية عمل حلقة التدريب، فراجع البرنامج التعليمي [Understanding pipelines, models and schedulers](../using-diffusers/write_own_pipeline) الذي يوضح نمط عملية إزالة التشويش الأساسية.

## إطلاق النص البرمجي

بمجرد إجراء جميع التغييرات أو إذا كنت راضيًا عن التكوين الافتراضي، فأنت مستعد لإطلاق نص التدريب البرمجي! 🚀

<Tip warning={true}>
تستغرق عملية التدريب الكاملة ساعتين على 4xV100 GPUs.
</Tip>

<hfoptions id="launchtraining">

لتشغيل التدريب على وحدة معالجة رسومات واحدة، استخدم ما يلي:

```bash
accelerate launch train_unconditional.py \
--dataset_name="huggan/flowers-102-categories" \
--output_dir="ddpm-ema-flowers-64" \
--mixed_precision="fp16" \
--push_to_hub
```

</hfoption>

إذا كنت تتدرب على أكثر من وحدة معالجة رسومات واحدة، فأضف المعلمة `--multi_gpu` إلى أمر التدريب:

```bash
accelerate launch --multi_gpu train_unconditional.py \
--dataset_name="huggan/flowers-102-categories" \
--output_dir="ddpm-ema-flowers-64" \
--mixed_precision="fp16" \
--push_to_hub
```

</hfoption>

</hfoptions>

ينشئ نص التدريب البرمجي ملف نقطة تفتيش ويحفظه في مستودعك. الآن يمكنك تحميل نموذجك المدرب واستخدامه للاستنتاج:

```py
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("anton-l/ddpm-butterflies-128").to("cuda")
image = pipeline().images[0]
```