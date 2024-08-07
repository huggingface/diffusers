بالتأكيد، سألتزم بالتعليمات المذكورة. فيما يلي ترجمة النص الموجود في الفقرات والعناوين:

# التقطير الاتساقي الكامن

تستطيع نماذج الاتساق الكامنة (LCMs) توليد صور عالية الجودة في بضع خطوات فقط، مما يمثل قفزة كبيرة إلى الأمام لأن العديد من الأنابيب تحتاج إلى 25 خطوة على الأقل. يتم إنتاج نماذج LCM من خلال تطبيق طريقة التقطير الاتساقي الكامن على أي نموذج Stable Diffusion. تعمل هذه الطريقة من خلال تطبيق التقطير الموجه أحادي المرحلة على المساحة الكامنة، ودمج طريقة "تخطي الخطوة" لتخطي الخطوات الزمنية باستمرار لتسريع عملية التقطير (راجع الأقسام 4.1 و4.2 و4.3 من الورقة لمزيد من التفاصيل).

إذا كنت تتدرب على وحدة معالجة رسومات (GPU) ذات ذاكرة وصول عشوائي (VRAM) محدودة، فجرّب تمكين "gradient_checkpointing" و"gradient_accumulation_steps" و"mixed_precision" لتقليل استخدام الذاكرة وتسريع التدريب. يمكنك تقليل استخدام الذاكرة أكثر من خلال تمكين الاهتمام الفعال من حيث الذاكرة مع [xFormers](../optimization/xformers) ومُحسّن 8 بت من [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).

سيتعمق هذا الدليل في دراسة نص البرنامج النصي [train_lcm_distill_sd_wds.py](https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/train_lcm_distill_sd_wds.py) لمساعدتك على التعرف عليه بشكل أفضل، وكيف يمكنك تكييفه مع حالتك الاستخدامية الخاصة.

قبل تشغيل البرنامج النصي، تأكد من تثبيت المكتبة من المصدر:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

بعد ذلك، انتقل إلى مجلد المثال الذي يحتوي على البرنامج النصي للتدريب وقم بتثبيت التبعيات المطلوبة للبرنامج النصي الذي تستخدمه:

```bash
cd examples/consistency_distillation
pip install -r requirements.txt
```

🤗 Accelerate هي مكتبة تساعدك على التدريب على وحدات معالجة رسومات (GPUs) أو وحدات معالجة الرسومات (TPUs) متعددة أو باستخدام الدقة المختلطة. سيقوم تلقائيًا بتكوين إعداد التدريب الخاص بك بناءً على أجهزتك وبيئتك. اطلع على الجولة السريعة من 🤗 Accelerate [Quick tour](https://huggingface.co/docs/accelerate/quicktour) لمعرفة المزيد.

قم بتهيئة بيئة 🤗 Accelerate (جرّب تمكين `torch.compile` لتسريع التدريب بشكل كبير):

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

أخيرًا، إذا كنت تريد تدريب نموذج على مجموعة بياناتك الخاصة، فراجع دليل [إنشاء مجموعة بيانات للتدريب](create_dataset) لمعرفة كيفية إنشاء مجموعة بيانات تعمل مع البرنامج النصي للتدريب.

## معلمات البرنامج النصي

تسلط الأقسام التالية الضوء على أجزاء من البرنامج النصي للتدريب والتي تُعد مهمة لفهم كيفية تعديلها، ولكنها لا تغطي كل جانب من جوانب البرنامج النصي بالتفصيل. إذا كنت مهتمًا بمعرفة المزيد، فلا تتردد في قراءة البرنامج النصي [script](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py) وأخبرنا إذا كان لديك أي أسئلة أو مخاوف.

يوفر البرنامج النصي للتدريب العديد من المعلمات لمساعدتك على تخصيص عملية تشغيل التدريب. يمكن العثور على جميع المعلمات ووصفها في دالة [`parse_args()`](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L419). توفر هذه الدالة قيمًا افتراضية لكل معلمة، مثل حجم دفعة التدريب ومعدل التعلم، ولكن يمكنك أيضًا تعيين قيمك الخاصة في أمر التدريب إذا أردت ذلك.

على سبيل المثال، لتسريع التدريب باستخدام الدقة المختلطة بتنسيق fp16، أضف المعلمة `--mixed_precision` إلى أمر التدريب:

```bash
accelerate launch train_lcm_distill_sd_wds.py \
--mixed_precision="fp16"
```

تتشابه معظم المعلمات مع المعلمات الموجودة في دليل التدريب [Text-to-image](text2image#script-parameters)، لذلك سيركز هذا الدليل على المعلمات ذات الصلة بالتقطير الاتساقي الكامن.

- `--pretrained_teacher_model`: المسار إلى نموذج التقطير الكامن المُدرب مسبقًا لاستخدامه كنموذج المعلم.
- `--pretrained_vae_model_name_or_path`: المسار إلى نموذج VAE مُدرب مسبقًا؛ من المعروف أن VAE الخاص بـ SDXL يعاني من عدم استقرار الأرقام، لذلك تسمح هذه المعلمة بتحديد VAE بديل (مثل هذا [VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) بواسطة madebyollin الذي يعمل في fp16).
- `--w_min` و `--w_max`: القيم الدنيا والقصوى لنطاق التوجيه لعينات نطاق التوجيه.
- `--num_ddim_timesteps`: عدد الخطوات الزمنية لعينات DDIM.
- `--loss_type`: نوع الخسارة (L2 أو Huber) لحسابها من أجل التقطير الاتساقي الكامن؛ تُفضل خسارة Huber بشكل عام لأنها أكثر مقاومة للقيم الشاذة.
- `--huber_c`: معلمة خسارة Huber.

## البرنامج النصي للتدريب

يبدأ البرنامج النصي للتدريب بإنشاء فئة مجموعة بيانات - [`Text2ImageDataset`](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L141) - لمعالجة الصور مسبقًا وإنشاء مجموعة بيانات للتدريب.

```py
def transform(example):
    image = example["image"]
    image = TF.resize(image, resolution, interpolation=transforms.InterpolationMode.BILINEAR)

    c_top, c_left, _, _ = transforms.RandomCrop.get_params(image, output_size=(resolution, resolution))
    image = TF.crop(image, c_top, c_left, resolution, resolution)
    image = TF.to_tensor(image)
    image = TF.normalize(image, [0.5], [0.5])

    example["image"] = image
    return example
```

لتحسين الأداء عند قراءة وكتابة مجموعات البيانات الكبيرة المخزنة في السحابة، يستخدم هذا البرنامج النصي تنسيق [WebDataset](https://github.com/webdataset/webdataset) لإنشاء خط أنابيب للمعالجة المسبقة لتطبيق التحويلات وإنشاء مجموعة بيانات ووحدة تغذية بيانات للتدريب. تتم معالجة الصور وإرسالها إلى حلقة التدريب دون الحاجة إلى تنزيل مجموعة البيانات بالكامل أولاً.

```py
processing_pipeline = [
    wds.decode("pil", handler=wds.ignore_and_continue),
    wds.rename(image="jpg;png;jpeg;webp", text="text;txt;caption", handler=wds.warn_and_continue),
    wds.map(filter_keys({"image", "text"})),
    wds.map(transform),
    wds.to_tuple("image", "text"),
]
```

في دالة [`main()`](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L768)، يتم تحميل جميع المكونات اللازمة مثل جدول مواعيد الضوضاء، والمحللات، ومشفرات النص، ونموذج VAE. يتم أيضًا تحميل شبكة UNet للمعلم هنا، وبعد ذلك يمكنك إنشاء شبكة UNet للطالب من شبكة UNet للمعلم. يتم تحديث شبكة UNet للطالب بواسطة المُحسّن أثناء التدريب.

```py
teacher_unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_teacher_model, subfolder="unet", revision=args.teacher_revision
)

unet = UNet2DConditionModel(**teacher_unet.config)
unet.load_state_dict(teacher_unet.state_dict(), strict=False)
unet.train()
```

الآن يمكنك إنشاء [المُحسّن](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L979) لتحديث معلمات UNet:

```py
optimizer = optimizer_class(
    unet.parameters(),
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)
```

قم بإنشاء [مجموعة البيانات](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L994):

```py
dataset = Text2ImageDataset(
    train_shards_path_or_url=args.train_shards_path_or_url,
    num_train_examples=args.max_train_samples,
    per_gpu_batch_size=args.train_batch_size,
    global_batch_size=args.train_batch_size * accelerator.num_processes,
    num_workers=args.dataloader_num_workers,
    resolution=args.resolution,
    shuffle_buffer_size=1000,
    pin_memory=True,
    persistent_workers=True,
)
train_dataloader = dataset.train_dataloader
```

بعد ذلك، أنت مستعد لإعداد [حلقة التدريب](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L1049) وتنفيذ طريقة التقطير الاتساقي الكامن (راجع الخوارزمية 1 في الورقة لمزيد من التفاصيل). يعتني هذا القسم من البرنامج النصي بإضافة الضوضاء إلى الكامنات، وأخذ العينات وإنشاء تضمين نطاق التوجيه، والتنبؤ بالصورة الأصلية من الضوضاء.

```py
pred_x_0 = predicted_origin(
    noise_pred,
    start_timesteps,
    noisy_model_input,
    noise_scheduler.config.prediction_type,
    alpha_schedule,
    sigma_schedule,
)

model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0
```

يحصل على [تنبؤات نموذج المعلم](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L1172) و [تنبؤات LCM](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L1209) بعد ذلك، ويحسب الخسارة، ثم يرجعها إلى الخلف إلى LCM.

```py
if args.loss_type == "l2":
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
elif args.loss_type == "huber":
    loss = torch.mean(
        torch.sqrt((model_pred.float() - target.float()) ** 2 + args.huber_c**2) - args.huber_c
    )
```

إذا كنت تريد معرفة المزيد حول كيفية عمل حلقة التدريب، فراجع الدليل التعليمي [Understanding pipelines, models and schedulers tutorial](../using-diffusers/write_own_pipeline) الذي يفكك النمط الأساسي لعملية إزالة التشويش.
## تشغيل السكربت 

الآن أنت مستعد لتشغيل سكربت التدريب والبدء في التقطير! 

للدليل، ستستخدم `--train_shards_path_or_url` لتحديد المسار إلى مجموعة بيانات [Conceptual Captions 12M](https://github.com/google-research-datasets/conceptual-12m) المخزنة على Hub [here](https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset). قم بتعيين متغير البيئة `MODEL_DIR` إلى اسم نموذج المعلم و`OUTPUT_DIR` إلى المكان الذي تريد حفظ النموذج فيه. 

```bash
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="path/to/saved/model"

accelerate launch train_lcm_distill_sd_wds.py \
--pretrained_teacher_model=$MODEL_DIR \
--output_dir=$OUTPUT_DIR \
--mixed_precision=fp16 \
--resolution=512 \
--learning_rate=1e-6 --loss_type="huber" --ema_decay=0.95 --adam_weight_decay=0.0 \
--max_train_steps=1000 \
--max_train_samples=4000000 \
--dataloader_num_workers=8 \
--train_shards_path_or_url="pipe:curl -L -s https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset/resolve/main/data/{00000..01099}.tar?download=true" \
--validation_steps=200 \
--checkpointing_steps=200 --checkpoints_total_limit=10 \
--train_batch_size=12 \
--gradient_checkpointing --enable_xformers_memory_efficient_attention \
--gradient_accumulation_steps=1 \
--use_8bit_adam \
--resume_from_checkpoint=latest \
--report_to=wandb \
--seed=453645634 \
--push_to_hub
``` 

بمجرد اكتمال التدريب، يمكنك استخدام LCM الجديد للاستنتاج. 

```py
from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler
import torch

unet = UNet2DConditionModel.from_pretrained("your-username/your-model", torch_dtype=torch.float16, variant="fp16")
pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", unet=unet, torch_dtype=torch.float16, variant="fp16")

pipeline.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipeline.to("cuda")

prompt = "sushi rolls in the form of panda heads, sushi platter"

image = pipeline(prompt, num_inference_steps=4, guidance_scale=1.0).images[0]
``` 

## LoRA 

LoRA هي تقنية تدريب لخفض عدد المعلمات القابلة للتدريب بشكل كبير. ونتيجة لذلك، يكون التدريب أسرع ويكون تخزين الأوزان الناتجة أسهل لأنها أصغر بكثير (~100MBs). استخدم [train_lcm_distill_lora_sd_wds.py](https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/train_lcm_distill_lora_sd_wds.py) أو [train_lcm_distill_lora_sdxl.wds.py](https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/train_lcm_distill_lora_sdxl_wds.py) لسكربت التدريب باستخدام LoRA. 

يناقش دليل [LoRA training](lora) تفاصيل سكربت التدريب باستخدام LoRA. 

## Stable Diffusion XL 

Stable Diffusion XL (SDXL) هو نموذج قوي للنص إلى الصورة يقوم بتوليد صور عالية الدقة، ويضيف مشفر نص ثانٍ إلى تصميمه. استخدم [train_lcm_distill_sdxl_wds.py](https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/train_lcm_distill_sdxl_wds.py) لسكربت التدريب لنموذج SDXL باستخدام LoRA. 

يناقش دليل [SDXL training](sdxl) تفاصيل سكربت التدريب لـ SDXL. 

## الخطوات التالية 

تهانينا على تقطير نموذج LCM! لمزيد من المعلومات حول LCM، قد يكون ما يلي مفيدًا: 

- تعلم كيفية استخدام [LCMs للاستنتاج](../using-diffusers/lcm) للنص إلى الصورة، والصورة إلى الصورة، ومع نقاط التحقق LoRA. 
- اقرأ منشور المدونة [SDXL in 4 steps with Latent Consistency LoRAs](https://huggingface.co/blog/lcm_lora) لمعرفة المزيد حول LCM-LoRA لـ SDXL للاستدلال السريع للغاية، ومقارنات الجودة، والمعايير، والمزيد.