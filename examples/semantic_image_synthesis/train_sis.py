# Documentation 
# https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline
import argparse
from ddpm_training import DDPMTrainingScheduler,DDPMScheduler
from vlb_loss import VLBLoss
from sis_dataset import SISDataset,CELEBAHQ_DICT

from diffusers.training_utils import EMAModel
from diffusers.models.unet_2d_sis import UNet2DSISModel,get_config
from diffusers.pipelines.semantic_only_diffusion.pipeline_semantic_only_diffusion import SemanticOnlyDiffusionPipeline
from diffusers.optimization import get_scheduler

from diffusers.utils.import_utils import is_bitsandbytes_available
from torch.utils.data import DataLoader,Subset
from datetime import datetime
import accelerate
from packaging import version
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import torch.nn.functional as F
import math
import shutil
import torch
import numpy as np
import os
from typing import List
from tqdm import tqdm

assert version.parse(accelerate.__version__) >= version.parse("0.16.0"),"Accelerate version should be higher than 0.16.0"

logger = get_logger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_img_dir',type=str,default='/mnt/c/BUSDATA/Datasets/CelebAMask-HQ/img/')
parser.add_argument('--dataset_ann_dir',type=str,default='/mnt/c/BUSDATA/Datasets/CelebAMask-HQ/mask/')
parser.add_argument('--dataset_img_size',type=int,default=128)
parser.add_argument('--output_dir',type=str,default='/mnt/c/BUSDATA/Datasets/CelebAMask-HQ/output/')
parser.add_argument('--train_batch_size',type=int,default=2)
parser.add_argument('--max_train_steps',type=int,default=5)
parser.add_argument('--ddp_variance_type',type=str,default='learned_range',
                    choices=['fixed_small','fixed_log','fixed_large','fixed_large_log','learned','learned_range'],)
parser.add_argument('--ddp_guidance_scale',type=float,default=1.5,
                    help='Unconditionnal guidance.')
parser.add_argument("--gradient_accumulation_steps",type=int,default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--lambda_vlb", type=float,default=1e-3)
parser.add_argument("--use_8bit_adam", action="store_true", 
                    help="Whether or not to use 8-bit Adam from bitsandbytes.")
parser.add_argument("--optim_weight_decay",type=float,default=1e-5,
    help='optimizer weight decay'
)
parser.add_argument("--optim_max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--learning_rate",type=float,default=1e-5,
    help="Initial learning rate (after the potential warmup period) to use.",
)
parser.add_argument("--scale_lr",action="store_true",default=True,
    help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
)
parser.add_argument("--lr_scheduler",type=str,default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
parser.add_argument("--lr_warmup_steps", type=int, default=500,
                    help="Number of steps for the warmup in the lr scheduler."
    )
parser.add_argument("--mixed_precision",type=str,
                    default='fp16',
                    choices=["no", "fp16", "bf16"],
                    help=(
                        "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                        " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                        " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
                    ))
parser.add_argument('--val_every_nepochs',type=int,default=5,help='Number of training epochs before validation...')
parser.add_argument('--val_num_samples',type=int,default=10)
parser.add_argument("--tracker_name",type=str,default=['mlflow','tensorboard'],
                    help=(
                        'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                        ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
                    ))
parser.add_argument("--checkpointing_steps",type=int,default=1000,
                    help="Indicate the checkpoint frequency")
parser.add_argument("--checkpointing_total_limit",type=int,default=10,
                    help="Indicate the max number of checkpoints to keep")
parser.add_argument("--resume_from_checkpoint",type=str,default="latest",
                    help="Indicate either the checkpoint directory or the latest")
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--use_ema',type=bool,default=True)
parser.add_argument('--debug',action='store_true',default=False)

def main(
        dataset_img_dir:str,
        dataset_ann_dir:str,
        dataset_img_size:int,
        output_dir:str,
        train_batch_size:int,
        max_train_steps:int,
        ddp_variance_type:str,
        ddp_guidance_scale:float,
        lambda_vlb:float,
        gradient_accumulation_steps:int,
        use_8bit_adam:bool,
        optim_weight_decay:float,
        optim_max_grad_norm:float,
        learning_rate:float,
        scale_lr:bool,
        lr_scheduler:str,
        lr_warmup_steps:int,
        mixed_precision:str,
        val_every_nepochs:int,
        val_num_samples:int,
        tracker_name:str,
        checkpointing_steps:int,
        checkpointing_total_limit:int,
        resume_from_checkpoint:str,
        seed:int,
        use_ema:bool,
        debug:bool=False
         ):
    # For debugging, we remove some examples...
    if debug:
        NMAX=100
    else:
        NMAX=None
    # We create the logging directory...
    logdir_name =  f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-SISModel"
    logdir_path = os.path.join(output_dir,logdir_name)
    
    # Configure Accelerate
    accelerator_project_configuration = ProjectConfiguration(project_dir=output_dir,logging_dir=logdir_path)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=tracker_name,
        project_config=accelerator_project_configuration,
    )
    if accelerator.is_main_process:
        os.makedirs(logdir_path,exist_ok=True)

    # Manage Seed
    set_seed(seed)
    # We create a Dataset and Dataloaders
    train_dataset = SISDataset(dataset_img_dir,dataset_ann_dir,img_size=dataset_img_size,cls_dict=CELEBAHQ_DICT,nmax=NMAX)

    indices = np.arange(len(train_dataset))
    np.random.shuffle(indices)
    train_indices = indices[val_num_samples:]
    val_indices = indices[:val_num_samples]


    train_dataloader = DataLoader(
        Subset(train_dataset,train_indices),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=min(os.cpu_count(),train_batch_size),
    )
    val_dataloader = DataLoader(
        Subset(train_dataset,val_indices),
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
    
    # We create the model
    # If the variance is learned, we'll use a model that output a 2xInput_Channel tensor.
    input_channels = 3
    if 'learned' in ddp_variance_type:
        learned_variance = True
        output_channels = 2*input_channels
    else:
        learned_variance = False
        output_channels = input_channels
    config = get_config(train_dataset.img_size,input_channels,output_channels,train_dataset.cls_count)
    unet = UNet2DSISModel(**config)
    if use_ema:
            ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DSISModel, model_config=unet.config)

    def save_model_hook(models:List[UNet2DSISModel], weights, output_dir):
        if accelerator.is_main_process:
            if use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()
    def load_model_hook(models:List[UNet2DSISModel], input_dir):
                if use_ema:
                    load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DSISModel)
                    ema_unet.load_state_dict(load_model.state_dict())
                    ema_unet.to(accelerator.device)
                    del load_model

                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()
                    # load diffusers style into model
                    load_model = UNet2DSISModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # We create the optimizer and scheduler
    if is_bitsandbytes_available() and use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    if scale_lr:
         # We rescale Lr with different parameters...
         learning_rate= learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
    optimizer = optimizer_class(
        unet.parameters(),
        lr=learning_rate,
        weight_decay=optim_weight_decay
    )

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )
    # Create Diffusion Items
    # We use a custom DDPM Scheduler dedicated for training...
    noise_scheduler = DDPMTrainingScheduler(variance_type=ddp_variance_type)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)


    global_step = 0
    first_epoch = 0
    # We prepare all of our items with Accelerate
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    if use_ema:
        ema_unet.to(accelerator.device)

    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0
    
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("train-sis")
    # We create the training loop
    for epoch in range(first_epoch, num_train_epochs):
        # Training Loop
        unet.train()
        total_loss=0.0
        simple_loss=0.0
        vlb_loss=0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                x,y,id = batch
                loss_total_batch=0.0
                # We create a noise like X
                noise = torch.randn_like(x)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (x.shape[0],), device=x.device)
                x_t = noise_scheduler.add_noise(x,noise,timesteps.long())
                if not learned_variance:
                    model_pred = unet.forward(x_t,timesteps,y).sample
                else:
                    # We predict the variance and we'll have to apply VLB Loss
                    model_output = unet.forward(x_t,timesteps,y).sample
                    # Here need to :
                    # Get predicted variance with DDPM Scheduler
                    model_pred,_,logvar_pred = noise_scheduler._get_p_mean_variance(model_output,timesteps)
                    # Get q variance with DDPM Scheduler
                    q_mean,_,q_logvar = noise_scheduler._get_q_mean_variance(timesteps,x,x_t)
                    kl_div = VLBLoss().forward(model_pred,logvar_pred,q_mean,q_logvar)
                    kl_div_mean = kl_div.reshape(x.shape[0],-1).mean(dim=1)
                    # We compute the VLB Loss
                    vlb_loss_batch = lambda_vlb*torch.where((timesteps>0),kl_div_mean,0.0).mean()
                    loss_total_batch+=vlb_loss_batch
                    vlb_loss += accelerator.gather(vlb_loss_batch.repeat(train_batch_size)).mean() / gradient_accumulation_steps

                # We compute the Loss_simple value...
                loss_simple_batch = F.mse_loss(model_pred.float(), noise.float(), reduction="none").mean()
                loss_total_batch += loss_simple_batch

                # We gather across devices for logging.
                simple_loss += accelerator.gather(loss_simple_batch.repeat(train_batch_size)).mean() / gradient_accumulation_steps
                total_loss += accelerator.gather(loss_total_batch.repeat(train_batch_size)).mean() / gradient_accumulation_steps

                accelerator.backward(loss_total_batch)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), optim_max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                logs = {"loss_simple": simple_loss.detach().item(),"loss_total": total_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                if learned_variance:
                    logs['loss_vlb']=vlb_loss.detach().item()
                global_step += 1
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                total_loss = 0.0
                simple_loss=0.0
                vlb_loss=0.0
                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if checkpointing_total_limit is not None:
                            checkpoints = os.listdir(output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= checkpointing_total_limit:
                                num_to_remove = len(checkpoints) - checkpointing_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    

            if global_step >= max_train_steps:
                break
        # Validation Loop
        if accelerator.is_main_process:
            images = []
            image_ids = []
            if  epoch % val_every_nepochs == 0:
                logger.info(
                    f"Running validation... \n Generating {val_num_samples} images"
                )
                val_scheduler = DDPMScheduler().from_config(noise_scheduler.config)
                # create pipeline
                pipeline = SemanticOnlyDiffusionPipeline(
                    unet=accelerator.unwrap_model(unet),
                    scheduler=val_scheduler
                )
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                for batch in tqdm(val_dataloader,desc='Validation'):
                    x,y,id = batch
                    x=x.unsqueeze(0)
                    y=y.unsqueeze(0)

                    generator = torch.Generator(device=accelerator.device).manual_seed(seed)
                    images.append(pipeline(segmap=y, num_inference_steps=50, guidance_scale=ddp_guidance_scale,generator=generator, eta=1.0).images[0])
                    image_ids.append(id)
                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "validation": [
                                    wandb.Image(image, caption=f"{i}: {image_ids[i]}")
                                    for i, image in enumerate(images)
                                ]
                            }
                        )
                del pipeline
                torch.cuda.empty_cache()
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unet:UNet2DSISModel
        unet.save_pretrained(output_dir)
        noise_scheduler.save_pretrained(output_dir)

if __name__ == '__main__':
    # Force main to have the rights arguments.
    # That's also good for Autocompletion.
    main(**vars(parser.parse_args()))