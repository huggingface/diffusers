import argparse
import gc
import os

import torch

from diffusers import AutoencoderKLLTX2Video
from diffusers.pipelines.ltx2 import LTX2ImageToVideoPipeline, LTX2LatentUpsamplePipeline, LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.utils import load_image


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="diffusers-internal-dev/new-ltx-model")
    parser.add_argument("--revision", type=str, default="main")

    parser.add_argument("--image_path", required=True, type=str)
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "An astronaut hatches from a fragile egg on the surface of the Moon, the shell cracking and peeling apart "
            "in gentle low-gravity motion. Fine lunar dust lifts and drifts outward with each movement, floating in "
            "slow arcs before settling back onto the ground. The astronaut pushes free in a deliberate, weightless "
            "motion, small fragments of the egg tumbling and spinning through the air. In the background, the deep "
            "darkness of space subtly shifts as stars glide with the camera's movement, emphasizing vast depth and "
            "scale. The camera performs a smooth, cinematic slow push-in, with natural parallax between the foreground "
            "dust, the astronaut, and the distant starfield. Ultra-realistic detail, physically accurate low-gravity "
            "motion, cinematic lighting, and a breath-taking, movie-like shot."
        ),
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion "
            "artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."
        ),
    )

    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--frame_rate", type=float, default=25.0)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply_scheduler_fix", action="store_true")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--vae_tiling", action="store_true")
    parser.add_argument("--use_video_latents", action="store_true")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="samples",
        help="Output directory for generated video",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="ltx2_i2v_video_upsampled.mp4",
        help="Filename of the exported generated video",
    )

    args = parser.parse_args()
    args.dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    return args


def main(args):
    pipeline = LTX2ImageToVideoPipeline.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=args.dtype,
    )
    if args.cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device=args.device)

    image = load_image(args.image_path)

    first_stage_output_type = "pil"
    if args.use_video_latents:
        first_stage_output_type = "latent"

    video, audio = pipeline(
        image=image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device=args.device).manual_seed(args.seed),
        output_type=first_stage_output_type,
        return_dict=False,
    )

    if args.use_video_latents:
        # Manually convert the audio latents to a waveform
        audio = audio.to(pipeline.audio_vae.dtype)
        audio = pipeline._denormalize_audio_latents(
            audio, pipeline.audio_vae.latents_mean, pipeline.audio_vae.latents_std
        )

        sampling_rate = pipeline.audio_sampling_rate
        hop_length = pipeline.audio_hop_length
        audio_vae_temporal_scale = pipeline.audio_vae_temporal_compression_ratio
        duration_s = args.num_frames / args.frame_rate
        latents_per_second = float(sampling_rate) / float(hop_length) / float(audio_vae_temporal_scale)
        audio_latent_frames = int(duration_s * latents_per_second)
        latent_mel_bins = pipeline.audio_vae.config.mel_bins // pipeline.audio_vae_mel_compression_ratio
        audio = pipeline._unpack_audio_latents(audio, audio_latent_frames, latent_mel_bins)

        audio = pipeline.audio_vae.decode(audio, return_dict=False)[0]
        audio = pipeline.vocoder(audio)

        # Get some pipeline configs for upsampling
        spatial_patch_size = pipeline.transformer_spatial_patch_size
        temporal_patch_size = pipeline.transformer_temporal_patch_size

    # upsample_pipeline = LTX2LatentUpsamplePipeline.from_pretrained(
    #     args.model_id, revision=args.revision, torch_dtype=args.dtype,
    # )
    output_sampling_rate = pipeline.vocoder.config.output_sampling_rate
    del pipeline  # Otherwise there might be an OOM error?
    torch.cuda.empty_cache()
    gc.collect()

    vae = AutoencoderKLLTX2Video.from_pretrained(
        args.model_id,
        subfolder="vae",
        revision=args.revision,
        torch_dtype=args.dtype,
    )
    latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
        args.model_id,
        subfolder="latent_upsampler",
        revision=args.revision,
        torch_dtype=args.dtype,
    )
    upsample_pipeline = LTX2LatentUpsamplePipeline(vae=vae, latent_upsampler=latent_upsampler)
    upsample_pipeline.to(device=args.device)
    if args.vae_tiling:
        upsample_pipeline.vae.enable_tiling()

    upsample_kwargs = {
        "height": args.height,
        "width": args.width,
        "output_type": "np",
        "return_dict": False,
    }
    if args.use_video_latents:
        upsample_kwargs["latents"] = video
        upsample_kwargs["num_frames"] = args.num_frames
        upsample_kwargs["spatial_patch_size"] = spatial_patch_size
        upsample_kwargs["temporal_patch_size"] = temporal_patch_size
    else:
        upsample_kwargs["video"] = video

    video = upsample_pipeline(**upsample_kwargs)[0]

    # Convert video to uint8 (but keep as NumPy array)
    video = (video * 255).round().astype("uint8")
    video = torch.from_numpy(video)

    encode_video(
        video[0],
        fps=args.frame_rate,
        audio=audio[0].float().cpu(),
        audio_sample_rate=output_sampling_rate,  # should be 24000
        output_path=os.path.join(args.output_dir, args.output_filename),
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
