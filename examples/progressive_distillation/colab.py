from dataclasses import dataclass


@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-butterflies-128"  # the model namy locally and on the HF Hub

    push_to_hub = True  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


config = TrainingConfig()

from datasets import load_dataset

config.dataset_name = "huggan/smithsonian_butterflies_subset"
dataset = load_dataset(config.dataset_name, split="train")
from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


dataset.set_transform(transform)
import torch
import os

from diffusers import UNet2DModel, DistillationPipeline, DDPMPipeline, DDPMScheduler
from accelerate import Accelerator

teacher = UNet2DModel.from_pretrained("bglick13/ddpm-butterflies-128", subfolder="unet")

# accelerator = Accelerator(
#     mixed_precision=config.mixed_precision,
#     gradient_accumulation_steps=config.gradient_accumulation_steps,
#     log_with="tensorboard",
#     logging_dir=os.path.join(config.output_dir, "logs"),
# )
# teacher = accelerator.prepare(teacher)
distiller = DistillationPipeline()
n_teacher_trainsteps = 1000
new_teacher, distilled_ema, distill_accelrator = distiller(
    teacher,
    n_teacher_trainsteps,
    dataset,
    epochs=100,
    batch_size=1,
    mixed_precision="fp16",
    sample_every=1,
    gamma=0.0,
    lr=0.3 * 5e-5,
)
new_scheduler = DDPMScheduler(num_train_timesteps=500, beta_schedule="squaredcos_cap_v2")
pipeline = DDPMPipeline(
    unet=distill_accelrator.unwrap_model(distilled_ema.averaged_model),
    scheduler=new_scheduler,
)

# run pipeline in inference (sample random noise and denoise)
images = pipeline(batch_size=4, output_type="numpy", generator=torch.manual_seed(0)).images

# denormalize the images and save to tensorboard
images_processed = (images * 255).round().astype("uint8")
from PIL import Image

img = Image.fromarray(images_processed[0])
img.save("denoised.png")
