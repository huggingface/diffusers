import argparse
import json
import os

import torch

from diffusers import DDIMPipeline, DDIMScheduler, DDPMPipeline
from PIL import Image


def _ddim_scheduler_from_ddpm_scheduler(sched):
    ret = DDIMScheduler(
        num_train_timesteps=sched.num_train_timesteps,
        trained_betas=sched.betas,
        clip_sample=sched.clip_sample,
    )
    assert torch.allclose(sched.alphas_cumprod, ret.alphas_cumprod)
    return ret


def main(args):
    pipeline = DDPMPipeline.from_pretrained(args.load_dir).to(
        torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")
    )

    if args.sampler == "ddim":
        ddimsched = _ddim_scheduler_from_ddpm_scheduler(pipeline.scheduler)
        pipeline = DDIMPipeline(pipeline.unet, ddimsched)

    num_steps = args.num_steps if args.num_steps != -1 else pipeline.scheduler.num_train_timesteps
    print(f"Number of timesteps used for sampling: {num_steps}")

    generator = torch.manual_seed(42)

    os.makedirs(os.path.join(args.load_dir, args.output_subdir), exist_ok=True)
    with open(os.path.join(args.load_dir, args.output_subdir, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)
    os.makedirs(os.path.join(args.load_dir, args.output_subdir, "imgs"), exist_ok=True)

    cnt = 0  # how many images generated so far
    numdigits = len(str(int(args.num_samples)))

    while cnt < args.num_samples:
        # run pipeline in inference (sample random noise and denoise)
        if args.sampler == "ddpm":
            images = pipeline(
                generator=generator,
                batch_size=args.batch_size,
                output_type="numpy",
            ).images
        elif args.sampler == "ddim":
            images = pipeline(
                generator=generator,
                batch_size=args.batch_size,
                output_type="numpy",
                num_inference_steps=args.num_steps,
                use_clipped_model_output=True,
                eta=args.ddim_eta,
            ).images

        # denormalize the images and save to tensorboard
        images_processed = (images * 255).round().astype("uint8")  # .transpose(0, 3, 1, 2)

        for img in list(images_processed):
            pilimage = Image.fromarray(img, mode="RGB")
            pilimage.save(
                os.path.join(args.load_dir, args.output_subdir, "imgs", f"{{:0{numdigits}d}}.png".format(cnt))
            )
            cnt += 1

        print(f"generated {cnt} images")

    print(f"generated {cnt} images and saved in {os.path.join(args.load_dir, args.output_subdir, 'imgs')}")

    print(pipeline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--load_dir", type=str, default="ddpm-model-64")
    parser.add_argument("--sampler", type=str, default="ddpm")  # can be ddpm or ddim
    parser.add_argument("--num_steps", type=int, default=-1)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_subdir", type=str, default="samples")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=-1)

    args = parser.parse_args()
    main(args)

