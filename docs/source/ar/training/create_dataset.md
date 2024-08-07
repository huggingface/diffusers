# إنشاء مجموعة بيانات للتدريب

إذا كنت لا تجد مجموعة البيانات التي تبحث عنها أو تريد استخدام مجموعة البيانات الخاصة بك، فيمكنك إنشاء مجموعة بيانات باستخدام مكتبة 🤗 [Datasets](hf.co/docs/datasets). يعتمد هيكل مجموعة البيانات على المهمة التي تريد تدريب نموذجك عليها. أبسط هيكل لمجموعة البيانات هو مجلد من الصور للمهام مثل توليد الصور غير المشروطة. وقد يكون هيكل مجموعة بيانات أخرى عبارة عن مجلد من الصور وملف نصي يحتوي على تعليقات نصية المقابلة للصور للمهام مثل توليد النص إلى الصورة.

سيوضح هذا الدليل طريقتين لإنشاء مجموعة بيانات للتدريب الدقيق:

- تقديم مجلد من الصور إلى وسيط `--train_data_dir`
- تحميل مجموعة بيانات إلى Hub وإمرار معرف مستودع مجموعة البيانات إلى وسيط `--dataset_name`

## تقديم مجموعة البيانات كمجلد

بالنسبة للتوليد غير المشروط، يمكنك تقديم مجموعة البيانات الخاصة بك كمجلد من الصور. يستخدم نص البرنامج النصي للتدريب الباني [`ImageFolder`](https://huggingface.co/docs/datasets/en/image_dataset#imagefolder) من 🤗 Datasets لإنشاء مجموعة بيانات تلقائيًا من المجلد. يجب أن يبدو هيكل الدليل الخاص بك على النحو التالي:

```bash
data_dir/xxx.png
data_dir/xxy.png
data_dir/[...]/xxz.png
```

مرر المسار إلى دليل مجموعة البيانات إلى وسيط `--train_data_dir`، ثم يمكنك بدء التدريب:

```bash
accelerate launch train_unconditional.py \
--train_data_dir <path-to-train-directory> \
<other-arguments>
```

## تحميل بياناتك إلى Hub

ابدأ بإنشاء مجموعة بيانات باستخدام ميزة [`ImageFolder`](https://huggingface.co/docs/datasets/image_load#imagefolder)، والتي تنشئ عمود `image` يحتوي على الصور المشفرة بواسطة PIL.

يمكنك استخدام معلمات `data_dir` أو `data_files` لتحديد موقع مجموعة البيانات. تدعم معلمة `data_files` تعيين ملفات محددة لمقاطع مجموعة البيانات مثل `train` أو `test`:

```python
from datasets import load_dataset

# example 1: local folder
dataset = load_dataset("imagefolder", data_dir="path_to_your_folder")

# example 2: local files (supported formats are tar, gzip, zip, xz, rar, zstd)
dataset = load_dataset("imagefolder", data_files="path_to_zip_file")

# example 3: remote files (supported formats are tar, gzip, zip, xz, rar, zstd)
dataset = load_dataset(
"imagefolder",
data_files="https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip",
)

# example 4: providing several splits
dataset = load_dataset(
"imagefolder", data_files={"train": ["path/to/file1", "path/to/file2"], "test": ["path/to/file3", "path/to/file4"]}
)
```

ثم استخدم طريقة [`~datasets.Dataset.push_to_hub`] لتحميل مجموعة البيانات إلى Hub:

```python
# assuming you have run the huggingface-cli login command in a terminal
dataset.push_to_hub("name_of_your_dataset")

# if you want to push to a private repo, simply pass private=True:
dataset.push_to_hub("name_of_your_dataset", private=True)
```

الآن، أصبحت مجموعة البيانات متاحة للتدريب عن طريق تمرير اسم مجموعة البيانات إلى وسيط `--dataset_name`:

```bash
accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
--dataset_name="name_of_your_dataset" \
<other-arguments>
```

## الخطوات التالية

الآن بعد أن قمت بإنشاء مجموعة بيانات، يمكنك توصيلها بوسيط `train_data_dir` (إذا كانت مجموعة البيانات المحلية الخاصة بك) أو `dataset_name` (إذا كانت مجموعة البيانات الخاصة بك على Hub) في نص البرنامج النصي للتدريب.

كخطوة تالية، لا تتردد في تجربة استخدام مجموعة البيانات الخاصة بك لتدريب نموذج للجيل غير المشروط أو [توليد النص إلى الصورة](text2image)!