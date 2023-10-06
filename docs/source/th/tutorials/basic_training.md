<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

[[open-in-colab]]

# ฝึก Diffusion Model

การสร้างภาพอย่างไรก็ตามโดยไม่ต้องมีเงื่อนไขเป็นการประยุกต์ใช้ที่นิยมของโมเดลการแพร่เผยที่สร้างภาพที่มีลักษณะเหมือนกับภาพในข้อมูลฝึกฝน โดยทั่วไปแล้วผลลัพธ์ที่ดีที่สุดนั้นได้มาจากการปรับแต่งโมเดลที่ได้รับการฝึกฝนล่วงหน้าบนชุดข้อมูลที่กำหนดเฉพาะ คุณสามารถค้นหาข้อมูลถ่ายครอบเหล่านี้ได้จาก [Hub](https://huggingface.co/search/full-text?q=unconditional-image-generation&type=model) แต่ถ้าคุณไม่พบอันใดที่คุณชอบ คุณสามารถฝึกฝนโมเดลของคุณเองเสมอได้!

บทนี้จะสอนคุณวิธีการฝึก [`UNet2DModel`] จากต้นฉบับบนชุดข้อมูลย่อยของ [Smithsonian Butterflies](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset) เพื่อสร้างผีเสื้อ 🦋 ของคุณเอง 🦋.

<Tip>

💡 บทนี้สอนการฝึกฝนที่ขึ้นอยู่กับ [Training with 🧨 Diffusers](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb) notebook สำหรับรายละเอียดเพิ่มเติมและบรรยากาศเกี่ยวกับโมเดลการแพร่เผยเช่นการทำงานของมัน โปรดตรวจสอบ notebook!

</Tip>

ก่อนที่คุณจะเริ่ม โปรดตรวจสอบว่าคุณได้ติดตั้ง 🤗 Datasets เพื่อโหลดและประมวลผลชุดข้อมูลภาพ และ 🤗 Accelerate เพื่อทำให้การฝึกฝนบนจำนวน GPU มีความเรียบง่าย คำสั่งต่อไปนี้จะติดตั้ง [TensorBoard](https://www.tensorflow.org/tensorboard) เพื่อแสดงแผนภูมิการฝึกฝน (คุณยังสามารถใช้ [Weights & Biases](https://docs.wandb.ai/) เพื่อติดตามการฝึกของคุณได้เช่นกัน).

```py
# uncomment to install the necessary libraries in Colab
#!pip install diffusers[training]
```

เรายังขอเสนอให้คุณแบ่งปันโมเดลของคุณกับชุมชน และเพื่อทำเช่นนั้น คุณต้องเข้าสู่ระบบบัญชี Hugging Face ของคุณ (สร้างได้ที่ [นี่](https://hf.co/join) หากยังไม่มีบัญชี!) คุณสามารถเข้าสู่ระบบจาก notebook และป้อนโทเคนของคุณเมื่อให้ในช่องที่ระบุ:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

หรือ login จาก terminal:

```bash
huggingface-cli login
```

ติดตั้ง [Git-LFS](https://git-lfs.com/):

```bash
!sudo apt -qq install git-lfs
!git config --global credential.helper store
```

## Training configuration

สร้าง`TrainingConfig` class สำหรับใส่ hyperparameters:

```py
>>> from dataclasses import dataclass


>>> @dataclass
... class TrainingConfig:
...     image_size = 128  # the generated image resolution
...     train_batch_size = 16
...     eval_batch_size = 16  # how many images to sample during evaluation
...     num_epochs = 50
...     gradient_accumulation_steps = 1
...     learning_rate = 1e-4
...     lr_warmup_steps = 500
...     save_image_epochs = 10
...     save_model_epochs = 30
...     mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
...     output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub

...     push_to_hub = True  # whether to upload the saved model to the HF Hub
...     hub_private_repo = False
...     overwrite_output_dir = True  # overwrite the old model when re-running the notebook
...     seed = 0


>>> config = TrainingConfig()
```

## โหลดชุดข้อมูล

คุณสามารถโหลดชุดข้อมูล [Smithsonian Butterflies](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset) ได้ง่ายๆ ด้วยไลบรารี 🤗 Datasets:

```py
>>> from datasets import load_dataset

>>> config.dataset_name = "huggan/smithsonian_butterflies_subset"
>>> dataset = load_dataset(config.dataset_name, split="train")
```

<Tip>

💡 คุณสามารถค้นหาชุดข้อมูลเพิ่มเติมจาก [HugGan Community Event](https://huggingface.co/huggan) หรือคุณสามารถใช้ชุดข้อมูลของคุณเองโดยการสร้าง [`ImageFolder`](https://huggingface.co/docs/datasets/image_dataset#imagefolder) ในเครื่องหลัก กำหนด `config.dataset_name` ให้เป็นรหัสที่อยู่ในเก็บข้อมูลถ้ามันมาจาก HugGan Community Event หรือ `imagefolder` หากคุณกำลังใช้รูปภาพของคุณเอง.

</Tip>

🤗 Datasets ใช้คุณลักษณะ [`~datasets.Image`] เพื่อถอดรหัสข้อมูลภาพโดยอัตโนมัติและโหลดมันเป็น [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html) ซึ่งเราสามารถแสดงผลได้:

```py
>>> import matplotlib.pyplot as plt

>>> fig, axs = plt.subplots(1, 4, figsize=(16, 4))
>>> for i, image in enumerate(dataset[:4]["image"]):
...     axs[i].imshow(image)
...     axs[i].set_axis_off()
>>> fig.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/butterflies_ds.png"/>
</div>

รูปภาพมีขนาดที่แตกต่างกันทั้งหมด ดังนั้นคุณจะต้องทำการประมวลผลก่อน:

* `Resize` เปลี่ยนขนาดรูปภาพเป็นขนาดที่กำหนดใน `config.image_size`
* `RandomHorizontalFlip` เพิ่มข้อมูลในชุดข้อมูลโดยการสะท้อนภาพแบบสุ่ม
* `Normalize` มีความสำคัญในการปรับขนาดค่าพิกเซลให้อยู่ในช่วง [-1, 1] ซึ่งเป็นสิ่งที่โมเดลต้องกา

```py
>>> from torchvision import transforms

>>> preprocess = transforms.Compose(
...     [
...         transforms.Resize((config.image_size, config.image_size)),
...         transforms.RandomHorizontalFlip(),
...         transforms.ToTensor(),
...         transforms.Normalize([0.5], [0.5]),
...     ]
... )
```

ใช้เมธอด [`~datasets.Dataset.set_transform`] ของ 🤗 Datasets เพื่อนำฟังก์ชัน `preprocess` มาใช้งานทันทีระหว่างการฝึกฝน:

```py
>>> def transform(examples):
...     images = [preprocess(image.convert("RGB")) for image in examples["image"]]
...     return {"images": images}


>>> dataset.set_transform(transform)
```

ตามใจคุณในการแสดงผลภาพอีกครั้งเพื่อยืนยันว่ามีการเปลี่ยนขนาดแล้ว ตอนนี้คุณพร้อมที่จะใส่ชุดข้อมูลลงใน [DataLoader](https://pytorch.org/docs/stable/data#torch.utils.data.DataLoader) เพื่อใช้ในการฝึกฝน!

```py
>>> import torch

>>> train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
```

## สร้าง UNet2DModel

โมเดลที่ได้รับการฝึกล่วงหน้าใน 🧨 Diffusers สามารถสร้างได้ง่ายๆ จากคลาสของโมเดลพรีเทรนด์พร้อมพารามิเตอร์ตามที่คุณต้องการ ตัวอย่างเช่น เพื่อสร้าง [`UNet2DModel`]:

```py
>>> from diffusers import UNet2DModel

>>> model = UNet2DModel(
...     sample_size=config.image_size,  # the target image resolution
...     in_channels=3,  # the number of input channels, 3 for RGB images
...     out_channels=3,  # the number of output channels
...     layers_per_block=2,  # how many ResNet layers to use per UNet block
...     block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
...     down_block_types=(
...         "DownBlock2D",  # a regular ResNet downsampling block
...         "DownBlock2D",
...         "DownBlock2D",
...         "DownBlock2D",
...         "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
...         "DownBlock2D",
...     ),
...     up_block_types=(
...         "UpBlock2D",  # a regular ResNet upsampling block
...         "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
...         "UpBlock2D",
...         "UpBlock2D",
...         "UpBlock2D",
...         "UpBlock2D",
...     ),
... )
```

มันเป็นไอเดียที่ดีที่จะตรวจสอบรูปร่างของภาพตัวอย่างว่าตรงกับรูปร่างผลลัพธ์ของโมเดลหรือไม่:

```py
>>> sample_image = dataset[0]["images"].unsqueeze(0)
>>> print("Input shape:", sample_image.shape)
Input shape: torch.Size([1, 3, 128, 128])

>>> print("Output shape:", model(sample_image, timestep=0).sample.shape)
Output shape: torch.Size([1, 3, 128, 128])
```

ดีมาก! ต่อไปคุณจะต้องใช้ตัวกำหนดเวลา (scheduler) เพื่อเพิ่ม noise ลงในภาพ

## สร้างตัวกำหนดเวลา (Scheduler)

ตัวกำหนดเวลา (scheduler) ทำงานต่างกันขึ้นอยู่กับว่าคุณกำลังใช้โมเดลสำหรับการฝึกฝนหรือการทำนาย ในขณะทำนาย ตัวกำหนดเวลาจะสร้างภาพจาก noise  ในขณะที่ฝึกฝน ตัวกำหนดเวลาจะนำผลลัพธ์จากโมเดล - หรือตัวอย่าง - จากจุดที่ระบุในขั้นตอนการแพร่เผยและปรับให้เกิด noise ในภาพตาม *ตารางการเกิด noise * และ *กฎการอัพเดท*.

เรามาดูที่ [`DDPMScheduler`] และใช้เมธอด `add_noise` เพื่อเพิ่ม noise สุ่มลงใน `sample_image` จากก่อนหน้านี้:

```py
>>> import torch
>>> from PIL import Image
>>> from diffusers import DDPMScheduler

>>> noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
>>> noise = torch.randn(sample_image.shape)
>>> timesteps = torch.LongTensor([50])
>>> noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

>>> Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/noisy_butterfly.png"/>
</div>

วัตถุประสงค์ในการฝึกฝนของโมเดลคือการทำนายเสียงที่เพิ่มเข้ากับภาพ ความสูญเสียในขั้นตอนนี้สามารถคำนวณได้โดย:

```py
>>> import torch.nn.functional as F

>>> noise_pred = model(noisy_image, timesteps).sample
>>> loss = F.mse_loss(noise_pred, noise)
```

## ฝึกโมเดล

ในขณะนี้, คุณมีส่วนใหญ่ขององค์ประกอบที่จำเป็นทั้งหมดเพื่อเริ่มต้นการฝึกโมเดลแล้วทุกอย่างที่เหลือนั้นคือการรวมทุกอย่างเข้าด้วยกัน.

ก่อนอื่น, คุณต้องมีตัวปรับโอพติไมซเซอร์ (optimizer) และตัวกำหนดอัตราการเรียนรู้ (learning rate scheduler):

```py
>>> from diffusers.optimization import get_cosine_schedule_with_warmup

>>> optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
>>> lr_scheduler = get_cosine_schedule_with_warmup(
...     optimizer=optimizer,
...     num_warmup_steps=config.lr_warmup_steps,
...     num_training_steps=(len(train_dataloader) * config.num_epochs),
... )
```

ต่อมา, คุณจะต้องมีวิธีการประเมินโมเดล สำหรับการประเมิน, คุณสามารถใช้ [`DDPMPipeline`] เพื่อสร้างชุดข้อมูล evaluation (batch) ของภาพตัวอย่างและบันทึกไว้ในรูปแบบตาราง:

```py
>>> from diffusers import DDPMPipeline
>>> from diffusers.utils import make_image_grid
>>> import math
>>> import os


>>> def evaluate(config, epoch, pipeline):
...     # Sample some images from random noise (this is the backward diffusion process).
...     # The default pipeline output type is `List[PIL.Image]`
...     images = pipeline(
...         batch_size=config.eval_batch_size,
...         generator=torch.manual_seed(config.seed),
...     ).images

...     # Make a grid out of the images
...     image_grid = make_image_grid(images, rows=4, cols=4)

...     # Save the images
...     test_dir = os.path.join(config.output_dir, "samples")
...     os.makedirs(test_dir, exist_ok=True)
...     image_grid.save(f"{test_dir}/{epoch:04d}.png")
```

ตอนนี้คุณสามารถรวบรวมทุกส่วนเหล่านี้เข้าด้วยกันในการวนซ้ำการฝึกฝนพร้อมกับ 🤗 Accelerate เพื่อให้สามารถบันทึก TensorBoard อย่างง่าย, การสะสม gradient, และการฝึกฝนด้วยความแม่นยำระดับผสม (mixed precision training) ในการอัปโหลดโมเดลไปยัง Hub, เขียนฟังก์ชันเพื่อรับชื่อที่เก็บของคุณและข้อมูล แล้วนำไปอัปโหลดไปยัง Hub

<Tip>

💡 การวนซ้ำการฝึกฝนด้านล่างอาจดูนานและท้าทาย แต่คุณจะได้รับผลตอบแทนเมื่อคุณเริ่มการฝึกฝนในบรรทัดเดียวเท่านั้น! หากคุณไม่สามารถรอและต้องการเริ่มสร้างภาพ โปรดรับสำเนาและเรียกใช้โค้ดด้านล่างนี้ได้เลย คุณสามารถกลับมาตรวจสอบการวนซ้ำการฝึกฝนในภายหลังได้เสมอ เช่นเมื่อคุณกำลังรอให้โมเดลเสร็จสิ้นการฝึกฝน 🤗

</Tip>

```py
>>> from accelerate import Accelerator
>>> from huggingface_hub import HfFolder, Repository, whoami
>>> from tqdm.auto import tqdm
>>> from pathlib import Path
>>> import os


>>> def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
...     if token is None:
...         token = HfFolder.get_token()
...     if organization is None:
...         username = whoami(token)["name"]
...         return f"{username}/{model_id}"
...     else:
...         return f"{organization}/{model_id}"


>>> def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
...     # Initialize accelerator and tensorboard logging
...     accelerator = Accelerator(
...         mixed_precision=config.mixed_precision,
...         gradient_accumulation_steps=config.gradient_accumulation_steps,
...         log_with="tensorboard",
...         project_dir=os.path.join(config.output_dir, "logs"),
...     )
...     if accelerator.is_main_process:
...         if config.push_to_hub:
...             repo_name = get_full_repo_name(Path(config.output_dir).name)
...             repo = Repository(config.output_dir, clone_from=repo_name)
...         elif config.output_dir is not None:
...             os.makedirs(config.output_dir, exist_ok=True)
...         accelerator.init_trackers("train_example")

...     # Prepare everything
...     # There is no specific order to remember, you just need to unpack the
...     # objects in the same order you gave them to the prepare method.
...     model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
...         model, optimizer, train_dataloader, lr_scheduler
...     )

...     global_step = 0

...     # Now you train the model
...     for epoch in range(config.num_epochs):
...         progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
...         progress_bar.set_description(f"Epoch {epoch}")

...         for step, batch in enumerate(train_dataloader):
...             clean_images = batch["images"]
...             # Sample noise to add to the images
...             noise = torch.randn(clean_images.shape).to(clean_images.device)
...             bs = clean_images.shape[0]

...             # Sample a random timestep for each image
...             timesteps = torch.randint(
...                 0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
...             ).long()

...             # Add noise to the clean images according to the noise magnitude at each timestep
...             # (this is the forward diffusion process)
...             noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

...             with accelerator.accumulate(model):
...                 # Predict the noise residual
...                 noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
...                 loss = F.mse_loss(noise_pred, noise)
...                 accelerator.backward(loss)

...                 accelerator.clip_grad_norm_(model.parameters(), 1.0)
...                 optimizer.step()
...                 lr_scheduler.step()
...                 optimizer.zero_grad()

...             progress_bar.update(1)
...             logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
...             progress_bar.set_postfix(**logs)
...             accelerator.log(logs, step=global_step)
...             global_step += 1

...         # After each epoch you optionally sample some demo images with evaluate() and save the model
...         if accelerator.is_main_process:
...             pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

...             if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
...                 evaluate(config, epoch, pipeline)

...             if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
...                 if config.push_to_hub:
...                     repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
...                 else:
...                     pipeline.save_pretrained(config.output_dir)
```

หืม มีโค้ดเยอะมากเลยนะ! แต่ตอนนี้คุณพร้อมที่จะเริ่มการฝึกฝนด้วยฟังก์ชัน [`~accelerate.notebook_launcher`] ของ 🤗 Accelerate แล้ว ให้ส่งโปรแกรมลูปการฝึกฝน อาร์กิวเมนต์การฝึกฝนทั้งหมด และจำนวนกระบวนการ (คุณสามารถเปลี่ยนค่านี้เป็นจำนวน GPU ที่คุณมี) ที่จะใช้สำหรับการฝึกฝน:

```py
>>> from accelerate import notebook_launcher

>>> args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

>>> notebook_launcher(train_loop, args, num_processes=1)
```

เมื่อการฝึกฝนเสร็จสิ้น ลองดูภาพ 🦋 ที่สร้างขึ้นโดยโมเดลการแพร่กระจายของคุณ!

```py
>>> import glob

>>> sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
>>> Image.open(sample_images[-1])
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/butterflies_final.png"/>
</div>

## ขั้นตอนต่อไป

การสร้างภาพโดยไม่มีเงื่อนไขเป็นหนึ่งในตัวอย่างของงานที่สามารถฝึกฝนได้ คุณสามารถสำรวจงานและเทคนิคการฝึกฝนอื่นๆ ได้โดยไปที่หน้า [🧨 ตัวอย่างการฝึก Diffusers](../training/overview) นี่คือตัวอย่างบางอย่างของสิ่งที่คุณสามารถเรียนรู้:

* [การกลับด้านข้อความ](../training/text_inversion), อัลกอริทึมที่สอนโมเดลความคิดทางภาพเฉพาะและรวมมันเข้ากับภาพที่สร้างขึ้น
* [DreamBooth](../training/dreambooth), เทคนิคสำหรับสร้างภาพส่วนบุคคลของเรื่องที่กำหนดโดยให้ภาพนำเข้าหลายภาพของเรื่องนั้น
* [Text2Image](../training/text2image) สู่การปรับแต่งโมเดล Stable Diffusion ด้วยชุดข้อมูลของคุณเอง
* [LoRA](../training/lora) ในการใช้ LoRA, เทคนิคที่ประหยัดหน่วยความจำสำหรับปรับแต่งโมเดลขนาดใหญ่อย่างรวดเร็วขึ้น
