# Wuerstchen

يقلل نموذج Wuerstchen بشكل كبير من التكاليف الحسابية عن طريق ضغط مساحة المخفية 42x، دون المساس بجودة الصورة وتسريع الاستدلال. أثناء التدريب، يستخدم Wuerstchen نموذجين (VQGAN + autoencoder) لضغط المخفية، ثم يتم شرط نموذج ثالث (نموذج انتشار المخفية المشروط بالنص) على هذه المساحة المضغوطة للغاية لتوليد صورة.

لتناسب النموذج السابق في ذاكرة GPU ولتسريع التدريب، جرّب تمكين `gradient_accumulation_steps` و`gradient_checkpointing` و`mixed_precision` على التوالي.

يستكشف هذا الدليل البرنامج النصي [train_text_to_image_prior.py](https://github.com/huggingface/diffusers/blob/main/examples/wuerstchen/text_to_image/train_text_to_image_prior.py) لمساعدتك على التعرف عليه بشكل أفضل، وكيف يمكنك تكييفه مع حالتك الاستخدام الخاص.

قبل تشغيل البرنامج النصي، تأكد من تثبيت المكتبة من المصدر:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

ثم انتقل إلى مجلد المثال الذي يحتوي على البرنامج النصي للتدريب وقم بتثبيت التبعيات المطلوبة للبرنامج النصي الذي تستخدمه:

```bash
cd examples/wuerstchen/text_to_image
pip install -r requirements.txt
```

🤗 Accelerate هي مكتبة للمساعدة في التدريب على وحدات GPU/TPUs متعددة أو باستخدام الدقة المختلطة. سيقوم تلقائيًا بتكوين إعداد التدريب الخاص بك بناءً على الأجهزة وبيئتك. الق نظرة على جولة 🤗 Accelerate [سريعة](https://huggingface.co/docs/accelerate/quicktour) لمعرفة المزيد.

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

تسلط الأقسام التالية الضوء على أجزاء من البرامج النصية للتدريب والتي تُعد مهمة لفهم كيفية تعديلها، ولكنها لا تغطي كل جانب من جوانب [البرنامج النصي](https://github.com/huggingface/diffusers/blob/main/examples/wuerstchen/text_to_image/train_text_to_image_prior.py) بالتفصيل. إذا كنت مهتمًا بمعرفة المزيد، فلا تتردد في قراءة البرامج النصية وإخبارنا إذا كان لديك أي أسئلة أو مخاوف.

## معلمات البرنامج النصي

يوفر البرنامج النصي للتدريب العديد من المعلمات لمساعدتك في تخصيص عملية تشغيل التدريب. توجد جميع المعلمات ووصفاتها في دالة [`parse_args()`](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/wuerstchen/text_to_image/train_text_to_image_prior.py#L192). يوفر قيم افتراضية لكل معلمة، مثل حجم دفعة التدريب ومعدل التعلم، ولكن يمكنك أيضًا تعيين قيمك الخاصة في أمر التدريب إذا كنت ترغب في ذلك.

على سبيل المثال، لزيادة سرعة التدريب باستخدام الدقة المختلطة بتنسيق fp16، أضف المعلمة `--mixed_precision` إلى أمر التدريب:

```bash
accelerate launch train_text_to_image_prior.py \
--mixed_precision="fp16"
```

تتشابه معظم المعلمات مع المعلمات الموجودة في دليل التدريب [Text-to-image](text2image#script-parameters)، لذلك دعونا نغوص مباشرة في البرنامج النصي لتدريب Wuerstchen!

## البرنامج النصي للتدريب

البرنامج النصي للتدريب مشابه أيضًا لدليل التدريب [Text-to-image](text2image#training-script)، ولكنه تم تعديله لدعم Wuerstchen. يركز هذا الدليل على التعليمات البرمجية الفريدة لبرنامج Wuerstchen النصي للتدريب.

تبدأ دالة [`main()`](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/wuerstchen/text_to_image/train_text_to_image_prior.py#L441) عن طريق تهيئة مشفر الصور - [EfficientNet](https://github.com/huggingface/diffusers/blob/main/examples/wuerstchen/text_to_image/modeling_efficient_net_encoder.py) - بالإضافة إلى الجدولة ومشغل الرموز المعتادين.

```py
with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
pretrained_checkpoint_file = hf_hub_download("dome272/wuerstchen", filename="model_v2_stage_b.pt")
state_dict = torch.load(pretrained_checkpoint_file, map_location="cpu")
image_encoder = EfficientNetEncoder()
image_encoder.load_state_dict(state_dict["effnet_state_dict"])
image_encoder.eval()
```

ستقوم أيضًا بتحميل نموذج [`WuerstchenPrior`] للتحسين.

```py
prior = WuerstchenPrior.from_pretrained(args.pretrained_prior_model_name_or_path, subfolder="prior")

optimizer = optimizer_cls(
prior.parameters(),
lr=args.learning_rate,
betas=(args.adam_beta1, args.adam_beta2),
weight_decay=args.adam_weight_decay,
eps=args.adam_epsilon,
)
```

بعد ذلك، ستطبق بعض [التحويلات](https://github.com/huggingface/diffusers/blob/65ef7a0c5c594b4f84092e328fbdd73183613b30/examples/wuerstchen/text_to_image/train_text_to_image_prior.py#L656) على الصور و[رموز](https://github.com/huggingface/diffusers/blob/65ef7a0c5c594b4f84092e328fbdd73183613b30/examples/wuerstchen/text_to_image/train_text_to_image_prior.py#L637) التعليقات التوضيحية:

```py
def preprocess_train(examples):
images = [image.convert("RGB") for image in examples[image_column]]
examples["effnet_pixel_values"] = [effnet_transforms(image) for image in images]
examples["text_input_ids"], examples["text_mask"] = tokenize_captions(examples)
return examples
```

أخيرًا، تتولى حلقة [التدريب](https://github.com/huggingface/diffusers/blob/65ef7a0c5c594b4f84092e328fbdd73183613b30/examples/wuerstchen/text_to_image/train_text_to_image_prior.py#L656) التعامل مع ضغط الصور إلى مساحة المخفية باستخدام `EfficientNetEncoder`، وإضافة ضوضاء إلى المخفية، والتنبؤ ببقايا الضوضاء باستخدام نموذج [`WuerstchenPrior`] .

```py
pred_noise = prior(noisy_latents, timesteps, prompt_embeds)
```

إذا كنت تريد معرفة المزيد حول كيفية عمل حلقة التدريب، فراجع البرنامج التعليمي [فهم الأنابيب والنماذج والمجدولين](../using-diffusers/write_own_pipeline) الذي يكسر النمط الأساسي لعملية إزالة التشويش.

## إطلاق البرنامج النصي

بمجرد إجراء جميع التغييرات أو موافقتك على التكوين الافتراضي، ستكون جاهزًا لإطلاق برنامج التدريب النصي! 🚀

قم بتعيين متغير البيئة `DATASET_NAME` إلى اسم مجموعة البيانات من Hub. يستخدم هذا الدليل مجموعة بيانات [Naruto BLIP captions](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions)، ولكن يمكنك إنشاء مجموعات بياناتك الخاصة والتدريب عليها أيضًا (راجع دليل [إنشاء مجموعة بيانات للتدريب](create_dataset)).

لمراقبة تقدم التدريب باستخدام Weights & Biases، أضف المعلمة `--report_to=wandb` إلى أمر التدريب. ستحتاج أيضًا إلى إضافة `--validation_prompt` إلى أمر التدريب لتتبع النتائج. يمكن أن يكون هذا مفيدًا جدًا في تصحيح أخطاء النموذج وعرض النتائج المتوسطة.

```bash
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch train_text_to_image_prior.py \
--mixed_precision="fp16" \
--dataset_name=$DATASET_NAME \
--resolution=768 \
--train_batch_size=4 \
--gradient_accumulation_steps=4 \
--gradient_checkpointing \
--dataloader_num_workers=4 \
--max_train_steps=15000 \
--learning_rate=1e-05 \
--max_grad_norm=1 \
--checkpoints_total_limit=3 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--validation_prompts="A robot naruto, 4k photo" \
--report_to="wandb" \
--push_to_hub \
--output_dir="wuerstchen-prior-naruto-model"
```

بمجرد اكتمال التدريب، يمكنك استخدام نموذجك المدرب حديثًا للاستدلال!

```py
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS

pipeline = AutoPipelineForText2Image.from_pretrained("path/to/saved/model", torch_dtype=torch.float16).to("cuda")

caption = "A cute bird naruto holding a shield"
images = pipeline(
caption,
width=1024,
height=1536,
prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
prior_guidance_scale=4.0,
num_images_per_prompt=2,
).images
```

## الخطوات التالية

تهانينا على تدريب نموذج Wuerstchen! لمعرفة المزيد حول كيفية استخدام نموذجك الجديد، قد يكون ما يلي مفيدًا:

- الق نظرة على وثائق [Wuerstchen](../api/pipelines/wuerstchen#text-to-image-generation) API لمعرفة المزيد حول كيفية استخدام الأنبوب لتوليد الصور النصية وقيوده.