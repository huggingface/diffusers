# Kandinsky 2.2

<Tip warning={true}>
هذا النص البرمجي تجريبي، ومن السهل أن ينحرف عن المسار الصحيح وأن يواجه مشكلات مثل النسيان الكارثي. جرّب استكشاف مختلف فرط المعلمات للحصول على أفضل النتائج في مجموعة بياناتك.
</Tip>

Kandinsky 2.2 هو نموذج متعدد اللغات للنص إلى الصورة قادر على إنتاج صور أكثر واقعية. يتضمن النموذج نموذجًا أوليًا للصورة لإنشاء تضمينات الصورة من موجهات النص، ونموذج فك تشفير يقوم بتوليد الصور بناءً على تضمينات النموذج الأولي. ولهذا السبب، ستجد نصين برمجيين منفصلين في Diffusers لـ Kandinsky 2.2، أحدهما لتدريب النموذج الأولي والآخر لتدريب نموذج فك التشفير. يمكنك تدريب كلا النموذجين بشكل منفصل، ولكن للحصول على أفضل النتائج، يجب تدريب كل من النماذج الأولية ونماذج فك التشفير.

اعتمادًا على وحدة معالجة الرسومات (GPU) لديك، قد تحتاج إلى تمكين `gradient_checkpointing` (⚠️ غير مدعوم للنموذج الأولي!)، و`mixed_precision`، و`gradient_accumulation_steps` للمساعدة في تكييف النموذج مع الذاكرة ولتسريع التدريب. يمكنك تقليل استخدام الذاكرة لديك بشكل أكبر عن طريق تمكين الاهتمام الفعال للذاكرة باستخدام [xFormers](فشلت الإصدار [v0.0.16](https://github.com/huggingface/diffusers/issues/2234#issuecomment-1416931212) في التدريب على بعض وحدات معالجة الرسومات (GPU)، لذا قد تحتاج إلى تثبيت إصدار التطوير بدلاً من ذلك).

يستكشف هذا الدليل النصين البرمجيين [train_text_to_image_prior.py](https://github.com/huggingface/diffusers/blob/main/examples/kandinsky2_2/text_to_image/train_text_to_image_prior.py) و [train_text_to_image_decoder.py](https://github.com/huggingface/diffusers/blob/main/examples/kandinsky2_2/text_to_image/train_text_to_image_decoder.py) لمساعدتك على التعرف عليه بشكل أفضل، وكيف يمكنك تكييفه مع حالتك الاستخدام الخاصة.

قبل تشغيل النصوص البرمجية، تأكد من تثبيت المكتبة من المصدر:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

بعد ذلك، انتقل إلى مجلد المثال الذي يحتوي على نص التدريب وقم بتثبيت التبعيات المطلوبة للنص البرمجي الذي تستخدمه:

```bash
cd examples/kandinsky2_2/text_to_image
pip install -r requirements.txt
```

<Tip>
🤗 Accelerate هي مكتبة للمساعدة في التدريب على وحدات معالجة الرسومات (GPU) / وحدات معالجة الرسومات (TPU) المتعددة أو باستخدام الدقة المختلطة. سيقوم تلقائيًا بتكوين إعداد التدريب الخاص بك بناءً على الأجهزة وبيئتك. الق نظرة على جولة سريعة من 🤗 Accelerate [Quick tour](https://huggingface.co/docs/accelerate/quicktour) لمعرفة المزيد.
</Tip>

قم بتهيئة بيئة 🤗 Accelerate:

```bash
accelerate config
```

لإعداد بيئة 🤗 Accelerate الافتراضية دون اختيار أي تكوينات:

```bash
accelerate config default
```

أو إذا لم تدعم بيئتك غلافًا تفاعليًا، مثل دفتر الملاحظات، فيمكنك استخدام ما يلي:

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

أخيرًا، إذا كنت تريد تدريب نموذج على مجموعة البيانات الخاصة بك، فراجع دليل [إنشاء مجموعة بيانات للتدريب] (create_dataset) لمعرفة كيفية إنشاء مجموعة بيانات تعمل مع نص التدريب.

<Tip>
تسلط الأقسام التالية الضوء على أجزاء من نصوص التدريب المهمة لفهم كيفية تعديلها، ولكنها لا تغطي كل جانب من جوانب النصوص البرمجية بالتفصيل. إذا كنت مهتمًا بمعرفة المزيد، فلا تتردد في قراءة النصوص البرمجية وأخبرنا إذا كان لديك أي أسئلة أو مخاوف.
</Tip>

## معلمات النص البرمجي

يوفر نص التدريب العديد من المعلمات لمساعدتك في تخصيص عملية تشغيل التدريب. يمكن العثور على جميع المعلمات ووصفاتها في وظيفة [`parse_args ()`](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_prior.py#L190). يوفر نص التدريب قيمًا افتراضية لكل معلمة، مثل حجم دفعة التدريب ومعدل التعلم، ولكن يمكنك أيضًا تعيين قيمك الخاصة في أمر التدريب إذا رغبت في ذلك.

على سبيل المثال، لتسريع التدريب باستخدام الدقة المختلطة بتنسيق fp16، أضف المعلمة `--mixed_precision` إلى أمر التدريب:

```bash
accelerate launch train_text_to_image_prior.py \
--mixed_precision="fp16"
```

تتشابه معظم المعلمات مع المعلمات الموجودة في دليل التدريب [Text-to-image](text2image#script-parameters)، لذلك دعنا ننتقل مباشرة إلى شرح نصوص تدريب Kandinsky!

### وزن SNR الأدنى

يمكن أن تساعد استراتيجية وزن [Min-SNR](https://huggingface.co/papers/2303.09556) في التدريب عن طريق إعادة توازن الخسارة لتحقيق تقارب أسرع. يدعم نص التدريب التنبؤ بـ `epsilon` (الضوضاء) أو `v_prediction`، ولكن Min-SNR متوافق مع كلا نوعي التنبؤ. استراتيجية الترجيح هذه مدعومة فقط بواسطة PyTorch وغير متوفرة في نص تدريب Flax.

أضف المعلمة `--snr_gamma` وقم بتعيينها على القيمة الموصى بها 5.0:

```bash
accelerate launch train_text_to_image_prior.py \
--snr_gamma=5.0
```

## نص التدريب

نص التدريب مشابه أيضًا لنص التدريب [Text-to-image](text2image#training-script)، ولكنه تم تعديله لدعم تدريب النماذج الأولية ونماذج فك التشفير. يركز هذا الدليل على التعليمات البرمجية الفريدة لنصوص تدريب Kandinsky 2.2.

<hfoptions id="script">
<hfoption id="prior model">
تحتوي وظيفة  [`main ()`](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_decoder.py#L440)  على التعليمات البرمجية لإعداد مجموعة البيانات وتدريب النموذج.

أحد الاختلافات الرئيسية التي ستلاحظها على الفور هو أن نص التدريب يحمل أيضًا [`~ transformers.CLIPImageProcessor`] - بالإضافة إلى جدول زمني ومعالج رموز - لمعالجة الصور ونموذج [`~ transformers.CLIPVisionModelWithProjection`] لترميز الصور:

```py
noise_scheduler = DDPMScheduler(beta_schedule="squaredcos_cap_v2", prediction_type="sample")
image_processor = CLIPImageProcessor.from_pretrained(
    args.pretrained_prior_model_name_or_path, subfolder="image_processor"
)
tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_prior_model_name_or_path, subfolder="tokenizer")

with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_prior_model_name_or_path, subfolder="image_encoder", torch_dtype=weight_dtype
    ).eval()
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_prior_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype
    ).eval()
```

يستخدم Kandinsky [`PriorTransformer`] لتوليد تضمينات الصورة، لذا ستحتاج إلى إعداد المحسن لتعلم معلمات وضع النموذج.

```py
prior = PriorTransformer.from_pretrained(args.pretrained_prior_model_name_or_path, subfolder="prior")
prior.train()
optimizer = optimizer_cls(
    prior.parameters(),
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)
```

بعد ذلك ، تتم معالجة تعليقات الإدخال بواسطة المعالج، وتتم [معالجة] الصور [preprocessed](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_prior.py#L632) بواسطة [`~ transformers.CLIPImageProcessor`]:

```py
def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["clip_pixel_values"] = image_processor(images, return_tensors="pt").pixel_values
    examples["text_input_ids"], examples["text_mask"] = tokenize_captions(examples)
    return examples
```

أخيرًا، تقوم حلقة التدريب [التدريب](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_prior.py#L718) بتحويل الصور المدخلة إلى بيانات كامنة، وإضافة ضوضاء إلى تضمينات الصورة، والتنبؤ بها:

```py
model_pred = prior(
    noisy_latents,
    timestep=timesteps,
    proj_embedding=prompt_embeds,
    encoder_hidden_states=text_encoder_hidden_states,
    attention_mask=text_mask,
).predicted_image_embedding
```

إذا كنت تريد معرفة المزيد حول كيفية عمل حلقة التدريب, فراجع البرنامج التعليمي [فهم الأنابيب والنماذج والمجدولين](../using-diffusers/write_own_pipeline) الذي يكسر النمط الأساسي لعملية إزالة التشويش.

</hfoption>
<hfoption id="decoder model">

تحتوي وظيفة [`main ()`](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_decoder.py#L440) على التعليمات البرمجية لإعداد مجموعة البيانات وتدريب النموذج.

على عكس النموذج الأولي، يقوم فك التشفير بتضمين [`VQModel`] لفك تشفير البيانات الكامنة إلى صور ويستخدم [`UNet2DConditionModel`]:

```py
with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
    vae = VQModel.from_pretrained(
        args.pretrained_decoder_model_name_or_path, subfolder="movq", torch_dtype=weight_dtype
    ).eval()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_prior_model_name_or_path, subfolder="image_encoder", torch_dtype=weight_dtype
    ).eval()
unet = UNet2DConditionModel.from_pretrained(args.pretrained_decoder_model_name_or_path, subfolder="unet")
```

بعد ذلك، يتضمن النص البرمجي العديد من تحويلات الصور ووظيفة [معالجة](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_decoder.py#L622) لتطبيق التحولات على الصور وإرجاع قيم البكسل:

```py
def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["clip_pixel_values"] = image_processor(images, return_tensors="pt").pixel_values
    return examples
```

أخيرًا، تتولى حلقة التدريب [التدريب](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_decoder.py#L706) التعامل مع تحويل الصور إلى بيانات، وإضافة ضوضاء، والتنبؤ ببقايا الضوضاء.

إذا كنت تريد معرفة المزيد حول كيفية عمل حلقة التدريب، فراجع البرنامج التعليمي [فهم الأنابيب والنماذج والمجدولين](../using-diffusers/write_own_pipeline) الذي يكسر النمط الأساسي لعملية إزالة التشويش.

```py
model_pred = unet(noisy_latents, timesteps, None, added_cond_kwargs=added_cond_kwargs).sample[:, :4]
```

</hfoption>
</hfoptions>

## تشغيل السكربت

عندما تنتهي من إجراء جميع التغييرات أو تكون راضيًا عن التكوين الافتراضي، فستكون جاهزًا لتشغيل سكربت التدريب! 🚀

ستقوم بالتدريب على مجموعة بيانات [Naruto BLIP captions](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) لإنشاء شخصيات Naruto الخاصة بك، ولكن يمكنك أيضًا إنشاء مجموعة البيانات الخاصة بك والتدريب عليها من خلال اتباع الدليل [إنشاء مجموعة بيانات للتدريب](create_dataset). قم بتعيين متغير البيئة `DATASET_NAME` إلى اسم مجموعة البيانات على المنصة أو إذا كنت تقوم بالتدريب على ملفاتك الخاصة، فقم بتعيين متغير البيئة `TRAIN_DIR` إلى مسار مجموعة البيانات الخاصة بك.

إذا كنت تقوم بالتدريب على أكثر من وحدة معالجة رسومية (GPU)، فقم بإضافة المعامل `--multi_gpu` إلى أمر `accelerate launch`.

<Tip>

لمراقبة تقدم التدريب باستخدام Weights & Biases، أضف المعامل `--report_to=wandb` إلى أمر التدريب. ستحتاج أيضًا إلى إضافة `--validation_prompt` إلى أمر التدريب لتتبع النتائج. يمكن أن يكون هذا مفيدًا جدًا لتصحيح أخطاء النموذج وعرض النتائج المتوسطة.

</Tip>

<hfoptions id="training-inference">

<hfoption id="prior model">

```bash
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image_prior.py \
  --dataset_name=$DATASET_NAME \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --validation_prompts="A robot naruto, 4k photo" \
  --report_to="wandb" \
  --push_to_hub \
  --output_dir="kandi2-prior-naruto-model"
```

</hfoption>

<hfoption id="decoder model">

```bash
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image_decoder.py \
  --dataset_name=$DATASET_NAME \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --validation_prompts="A robot naruto, 4k photo" \
  --report_to="wandb" \
  --push_to_hub \
  --output_dir="kandi2-decoder-naruto-model"
```

</hfoption>

</hfoptions>

بمجرد الانتهاء من التدريب، يمكنك استخدام النموذج الذي قمت بتدريبه للتنبؤ!

<hfoptions id="training-inference">

<hfoption id="prior model">

```py
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
import torch

prior_pipeline = DiffusionPipeline.from_pretrained(output_dir, torch_dtype=torch.float16)
prior_components = {"prior_" + k: v for k,v in prior_pipeline.components.items()}
pipeline = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", **prior_components, torch_dtype=torch.float16)

pipe.enable_model_cpu_offload()
prompt="A robot naruto, 4k photo"
image = pipeline(prompt=prompt, negative_prompt=negative_prompt).images[0]
```

<Tip>

لا تتردد في استبدال `kandinsky-community/kandinsky-2-2-decoder` بنقطة تفتيش فك التشفير المدربة الخاصة بك!

</Tip>

</hfoption>

<hfoption id="decoder model">

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("path/to/saved/model", torch_dtype=torch.float16)
pipeline.enable_model_cpu_offload()

prompt="A robot naruto, 4k photo"
image = pipeline(prompt=prompt).images[0]
```

بالنسبة لنموذج فك التشفير، يمكنك أيضًا إجراء التنبؤ من نقطة حفظ محفوظة، والتي يمكن أن تكون مفيدة لعرض النتائج المتوسطة. في هذه الحالة، قم بتحميل نقطة التفتيش في UNet:

```py
from diffusers import AutoPipelineForText2Image, UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained("path/to/saved/model" + "/checkpoint-<N>/unet")

pipeline = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", unet=unet, torch_dtype=torch.float16)
pipeline.enable_model_cpu_offload()

image = pipeline(prompt="A robot naruto, 4k photo").images[0]
```

</hfoption>

</hfoptions>

## الخطوات التالية

تهانينا على تدريب نموذج Kandinsky 2.2! لمعرفة المزيد عن كيفية استخدام نموذجك الجديد، قد تكون الأدلة التالية مفيدة:

- اقرأ دليل [Kandinsky](../using-diffusers/kandinsky) لمعرفة كيفية استخدامه في مجموعة متنوعة من المهام المختلفة (النص إلى الصورة، والصورة إلى الصورة، والإكمال، والتشابك)، وكيف يمكن دمجه مع ControlNet.
- اطلع على أدلة التدريب [DreamBooth](dreambooth) و [LoRA](lora) لمعرفة كيفية تدريب نموذج Kandinsky شخصي باستخدام بضع صور مثال فقط. يمكن حتى دمج تقنيتي التدريب هاتين!