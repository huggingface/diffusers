# InstructPix2Pix

[InstructPix2Pix](https://hf.co/papers/2211.09800) هو نموذج Stable Diffusion تم تدريبه على تعديل الصور بناءً على تعليمات بشرية. على سبيل المثال، يمكن أن يكون موجهك "اجعل السحب ممطرة" وسيعدل النموذج صورة الإدخال وفقًا لذلك. يتم ضبط هذا النموذج بناءً على موجه النص (أو تعليمات التحرير) وصورة الإدخال.

سيتعمق هذا الدليل في دراسة [train_instruct_pix2pix.py](https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix.py) Script التدريب لمساعدتك على التعرف عليه، وكيف يمكنك تكييفه مع حالتك الاستخدام.

قبل تشغيل البرنامج النصي، تأكد من تثبيت المكتبة من المصدر:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

بعد ذلك، انتقل إلى مجلد المثال الذي يحتوي على البرنامج النصي للتدريب وقم بتثبيت التبعيات المطلوبة للبرنامج النصي الذي تستخدمه:

```bash
cd examples/instruct_pix2pix
pip install -r requirements.txt
```

🤗 Accelerate هي مكتبة تساعدك على التدريب على وحدات GPU/TPU متعددة أو باستخدام الدقة المختلطة. سيقوم تلقائيًا بتكوين إعداد التدريب الخاص بك بناءً على الأجهزة وبيئتك. الق نظرة على 🤗 تسريع [جولة سريعة](https://huggingface.co/docs/accelerate/quicktour) لمعرفة المزيد.

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

أخيرًا، إذا كنت تريد تدريب نموذج على مجموعة البيانات الخاصة بك، فراجع دليل [إنشاء مجموعة بيانات للتدريب](create_dataset) لمعرفة كيفية إنشاء مجموعة بيانات تعمل مع البرنامج النصي للتدريب.

تسلط الأقسام التالية الضوء على أجزاء من البرنامج النصي للتدريب والتي تعد مهمة لفهم كيفية تعديلها، ولكنها لا تغطي كل جانب من جوانب البرنامج النصي بالتفصيل. إذا كنت مهتمًا بمعرفة المزيد، فلا تتردد في قراءة البرنامج النصي [هنا](https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix.py) ودعنا نعرف إذا كان لديك أي أسئلة أو مخاوف.

## معلمات البرنامج النصي

يحتوي البرنامج النصي للتدريب على العديد من المعلمات لمساعدتك في تخصيص عملية تشغيل التدريب. يمكن العثور على جميع المعلمات ووصفاتها في دالة [`parse_args()`](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/instruct_pix2pix/train_instruct_pix2pix.py#L65). يتم توفير القيم الافتراضية لمعظم المعلمات التي تعمل بشكل جيد جدًا، ولكن يمكنك أيضًا تعيين قيمك الخاصة في أمر التدريب إذا كنت تريد ذلك.

على سبيل المثال، لزيادة دقة صورة الإدخال:

```bash
accelerate launch train_instruct_pix2pix.py \
--resolution=512 \
```

تم وصف العديد من المعلمات الأساسية والمهمة في دليل تدريب [Text-to-image](text2image#script-parameters)، لذلك يركز هذا الدليل فقط على المعلمات ذات الصلة بـ InstructPix2Pix:

- `--original_image_column`: الصورة الأصلية قبل إجراء التعديلات
- `--edited_image_column`: الصورة بعد إجراء التعديلات
- `--edit_prompt_column`: التعليمات لتعديل الصورة
- `--conditioning_dropout_prob`: احتمال إسقاط الصور المعدلة وإسقاطات التحرير أثناء التدريب الذي يمكّن التوجيه الخالي من التصنيف (CFG) لإدخال أو كليهما

## البرنامج النصي للتدريب

يمكن العثور على تعليمات ما قبل المعالجة التعليمية وحلقة التدريب في دالة [`main()`](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/instruct_pix2pix/train_instruct_pix2pix.py#L374). هنا ستقوم بإجراء تغييراتك على البرنامج النصي للتدريب لتكييفه مع حالتك الاستخدام.

كما هو الحال مع معلمات البرنامج النصي، يتم توفير دليل تفصيلي للبرنامج النصي للتدريب في دليل تدريب [Text-to-image](text2image#training-script). بدلاً من ذلك، يلقي هذا الدليل نظرة على أجزاء البرنامج النصي ذات الصلة بـ InstructPix2Pix.

يبدأ البرنامج النصي بتعديل [عدد قنوات الإدخال](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/instruct_pix2pix/train_instruct_pix2pix.py#L445) في الطبقة التلافيفية الأولى من UNet لمراعاة صورة الشرط الإضافية لـ InstructPix2Pix:

```py
in_channels = 8
out_channels = unet.conv_in.out_channels
unet.register_to_config(in_channels=in_channels)

with torch.no_grad():
	new_conv_in = nn.Conv2d(
	in_channels، out_channels، kernel_size الخاص بـ unet.conv_in، خطوة unet.conv_in، حشو unet.conv_in
	)
	new_conv_in.weight.zero_()
	new_conv_in.weight [:،: 4،:،:].نسخ (unet.conv_in.weight)
	unet.conv_in = new_conv_in
```

يتم [تحديث](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/instruct_pix2pix/train_instruct_pix2pix.py#L545C1-L551C6) معلمات UNet هذه بواسطة المحسن:

```py
optimizer = optimizer_cls(
	unet.parameters()،
	lr=args.learning_rate،
	betas=(args.adam_beta1، args.adam_beta2)،
	weight_decay=args.adam_weight_decay،
	eps=args.adam_epsilon،
)
```

بعد ذلك، يتم [معالجة](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/instruct_pix2pix/train_instruct_pix2pix.py#L624) الصور المعدلة وتعليمات التحرير و [رموزها](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/instruct_pix2pix/train_instruct_pix2pix.py#L610C24-L610C24). من المهم تطبيق نفس تحويلات الصورة على الصور الأصلية والمعدلة.

```py
def preprocess_train(examples):
    preprocessed_images = preprocess_images(examples)

    original_images, edited_images = preprocessed_images.chunk(2)
    original_images = original_images.reshape(-1, 3, args.resolution, args.resolution)
    edited_images = edited_images.reshape(-1, 3, args.resolution, args.resolution)

    examples["original_pixel_values"] = original_images
    examples["edited_pixel_values"] = edited_images

    captions = list(examples[edit_prompt_column])
    examples["input_ids"] = tokenize_captions(captions)
    return examples
```

أخيرًا، في [حلقة التدريب](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/instruct_pix2pix/train_instruct_pix2pix.py#L730)، يبدأ بتشفير الصور المعدلة في مساحة خفية:

```py
latents = vae.encode(batch ["edited_pixel_values"].to(weight_dtype)).latent_dist.sample()
latents = latents * vae.config.scaling_factor
```

بعد ذلك، يقوم البرنامج النصي بتطبيق الإسقاط على تضمين صورة الإدخال وتعليمات التحرير لدعم CFG. وهذا ما يمكّن النموذج من تعديل تأثير تعليمات التحرير وصورة الإدخال على الصورة المعدلة.

```py
encoder_hidden_states = text_encoder(batch["input_ids"])[0]
original_image_embeds = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()

if args.conditioning_dropout_prob is not None:
    random_p = torch.rand(bsz, device=latents.device, generator=generator)
    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
    null_conditioning = text_encoder(tokenize_captions([""]).to(accelerator.device))[0]
    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

    image_mask_dtype = original_image_embeds.dtype
    image_mask = 1 - (
        (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
    )
    image_mask = image_mask.reshape(bsz, 1, 1, 1)
    original_image_embeds = image_mask * original_image_embeds
```


هذا كل شيء! وبصرف النظر عن الاختلافات الموضحة هنا، فإن بقية البرنامج النصي مشابه جدًا لبرنامج نصي تدريب [Text-to-image](text2image#training-script)، لذا لا تتردد في الاطلاع عليه للحصول على مزيد من التفاصيل. إذا كنت تريد معرفة المزيد حول كيفية عمل حلقة التدريب، فراجع البرنامج التعليمي [فهم الأنابيب والنماذج والمجدولين](../using-diffusers/write_own_pipeline) الذي يكسر النمط الأساسي لعملية إزالة التشويش.

## إطلاق البرنامج النصي

بمجرد أن تشعر بالرضا عن التغييرات التي أجريتها على البرنامج النصي أو إذا كنت راضيًا عن التكوين الافتراضي، فأنت مستعد لإطلاق البرنامج النصي للتدريب! 🚀

يستخدم هذا الدليل مجموعة بيانات [fusing/instructpix2pix-1000-samples](https://huggingface.co/datasets/fusing/instructpix2pix-1000-samples)، والتي تعد إصدارًا أصغر من [مجموعة البيانات الأصلية](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered). يمكنك أيضًا إنشاء مجموعة البيانات الخاصة بك واستخدامها إذا كنت تريد ذلك (راجع دليل [إنشاء مجموعة بيانات للتدريب](create_dataset)).

قم بتعيين متغير البيئة `MODEL_NAME` إلى اسم النموذج (يمكن أن يكون معرف النموذج على Hub أو مسارًا إلى نموذج محلي)، و`DATASET_ID` إلى اسم مجموعة البيانات على Hub. يقوم البرنامج النصي بإنشاء جميع المكونات (مستخرج الميزات، والمجدول، ومشفر النص، وUNet، إلخ) وحفظها في مجلد فرعي في مستودعك.

للحصول على نتائج أفضل، جرب عمليات تشغيل التدريب الأطول باستخدام مجموعة بيانات أكبر. لقد قمنا فقط باختبار هذا البرنامج النصي للتدريب على مجموعة بيانات صغيرة الحجم.

لمراقبة تقدم التدريب باستخدام Weights and Biases، أضف المعلمة `--report_to=wandb` إلى أمر التدريب وحدد صورة التحقق من الصحة باستخدام `--val_image_url` وطلب التحقق من الصحة باستخدام `--validation_prompt`. يمكن أن يكون هذا مفيدًا جدًا في تصحيح أخطاء النموذج.

إذا كنت تتدرب على أكثر من وحدة GPU واحدة، فأضف المعلمة `--multi_gpu` إلى أمر `accelerate launch`.

```bash
accelerate launch --mixed_precision="fp16" train_instruct_pix2pix.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--dataset_name=$DATASET_ID \
--enable_xformers_memory_efficient_attention \
--resolution=256 \
--random_flip \
--train_batch_size=4 \
--gradient_accumulation_steps=4 \
--gradient_checkpointing \
--max_train_steps=15000 \
--checkpointing_steps=5000 \
--checkpoints_total_limit=1 \
--learning_rate=5e-05 \
--max_grad_norm=1 \
--lr_warmup_steps=0 \
--conditioning_dropout_prob=0.05 \
--mixed_precision=fp16 \
--seed=42 \
--push_to_hub
```

بعد الانتهاء من التدريب، يمكنك استخدام InstructPix2Pix الجديد للتنبؤ:


```py
import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image

pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained("your_cool_model", torch_dtype=torch.float16).to("cuda")
generator = torch.Generator("cuda").manual_seed(0)

image = load_image("https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/test_pix2pix_4.png")
prompt = "add some ducks to the lake"
num_inference_steps = 20
image_guidance_scale = 1.5
guidance_scale = 10

edited_image = pipeline(
   prompt,
   image=image,
   num_inference_steps=num_inference_steps,
   image_guidance_scale=image_guidance_scale,
   guidance_scale=guidance_scale,
   generator=generator,
).images[0]
edited_image.save("edited_image.png")
```


يجب عليك تجربة قيم مختلفة لـ `num_inference_steps`، و`image_guidance_scale`، و`guidance_scale` لمعرفة كيفية تأثيرها على سرعة التنبؤ وجودته. تؤثر معلمات مقياس التوجيه بشكل كبير على النموذج لأنها تتحكم في مدى تأثير صورة الإدخال وتعليمات التحرير على الصورة المعدلة.

## Stable Diffusion XL

Stable Diffusion XL (SDXL) هو نموذج قوي لتوليد الصور النصية ينشئ صورًا عالية الدقة، ويضيف مشفر نص ثانٍ إلى بنيتها. استخدم البرنامج النصي للتدريب [`train_instruct_pix2pix_sdxl.py`](https://github.com/huggingface/diffusers/blob/