
import argparse
import torch

from diffusers.pipelines.bddm import DiffWave, BDDMPipeline
from diffusers import DDPMScheduler


def convert_bddm_orginal(checkpoint_path, noise_scheduler_checkpoint_path, output_path):
    sd = torch.load(checkpoint_path, map_location="cpu")["model_state_dict"]
    noise_scheduler_sd = torch.load(noise_scheduler_checkpoint_path, map_location="cpu")

    model = DiffWave()
    model.load_state_dict(sd, strict=False)

    ts, _, betas, _ = noise_scheduler_sd
    ts, betas = list(ts.numpy().tolist()), list(betas.numpy().tolist())

    noise_scheduler = DDPMScheduler(
        timesteps=12,
        trained_betas=betas,
        timestep_values=ts,
        clip_sample=False,
        tensor_format="np",
    )

    pipeline = BDDMPipeline(model, noise_scheduler)
    pipeline.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--noise_scheduler_checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    convert_bddm_orginal(args.checkpoint_path, args.noise_scheduler_checkpoint_path, args.output_path)


