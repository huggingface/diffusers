import argparse
import os
from fractions import Fraction
from typing import Optional

import av  # Needs to be installed separately (`pip install av`)
import torch
from PIL import Image

from diffusers.pipelines.ltx2 import LTX2ImageToVideoPipeline


# Video export functions copied from original LTX 2.0 code
def _prepare_audio_stream(container: av.container.Container, audio_sample_rate: int) -> av.audio.AudioStream:
    """
    Prepare the audio stream for writing.
    """
    audio_stream = container.add_stream("aac", rate=audio_sample_rate)
    audio_stream.codec_context.sample_rate = audio_sample_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, audio_sample_rate)
    return audio_stream


def _resample_audio(
    container: av.container.Container, audio_stream: av.audio.AudioStream, frame_in: av.AudioFrame
) -> None:
    cc = audio_stream.codec_context

    # Use the encoder's format/layout/rate as the *target*
    target_format = cc.format or "fltp"  # AAC → usually fltp
    target_layout = cc.layout or "stereo"
    target_rate = cc.sample_rate or frame_in.sample_rate

    audio_resampler = av.audio.resampler.AudioResampler(
        format=target_format,
        layout=target_layout,
        rate=target_rate,
    )

    audio_next_pts = 0
    for rframe in audio_resampler.resample(frame_in):
        if rframe.pts is None:
            rframe.pts = audio_next_pts
        audio_next_pts += rframe.samples
        rframe.sample_rate = frame_in.sample_rate
        container.mux(audio_stream.encode(rframe))

    # flush audio encoder
    for packet in audio_stream.encode():
        container.mux(packet)


def _write_audio(
    container: av.container.Container,
    audio_stream: av.audio.AudioStream,
    samples: torch.Tensor,
    audio_sample_rate: int,
) -> None:
    if samples.ndim == 1:
        samples = samples[:, None]

    if samples.shape[1] != 2 and samples.shape[0] == 2:
        samples = samples.T

    if samples.shape[1] != 2:
        raise ValueError(f"Expected samples with 2 channels; got shape {samples.shape}.")

    # Convert to int16 packed for ingestion; resampler converts to encoder fmt.
    if samples.dtype != torch.int16:
        samples = torch.clip(samples, -1.0, 1.0)
        samples = (samples * 32767.0).to(torch.int16)

    frame_in = av.AudioFrame.from_ndarray(
        samples.contiguous().reshape(1, -1).cpu().numpy(),
        format="s16",
        layout="stereo",
    )
    frame_in.sample_rate = audio_sample_rate

    _resample_audio(container, audio_stream, frame_in)


def encode_video(
    video: torch.Tensor, fps: int, audio: Optional[torch.Tensor], audio_sample_rate: Optional[int], output_path: str
) -> None:
    video_np = video.cpu().numpy()

    _, height, width, _ = video_np.shape

    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    if audio is not None:
        if audio_sample_rate is None:
            raise ValueError("audio_sample_rate is required when audio is provided")

        audio_stream = _prepare_audio_stream(container, audio_sample_rate)

    for frame_array in video_np:
        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    if audio is not None:
        _write_audio(container, audio_stream, audio, audio_sample_rate)

    container.close()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="diffusers-internal-dev/new-ltx-model")
    parser.add_argument("--revision", type=str, default="main")

    parser.add_argument("--image_path", required=True, type=str)
    parser.add_argument(
        "--prompt",
        type=str,
        default="An astronaut hatches from a fragile egg on the surface of the Moon, the shell cracking and peeling apart in gentle low-gravity motion. Fine lunar dust lifts and drifts outward with each movement, floating in slow arcs before settling back onto the ground. The astronaut pushes free in a deliberate, weightless motion, small fragments of the egg tumbling and spinning through the air. In the background, the deep darkness of space subtly shifts as stars glide with the camera's movement, emphasizing vast depth and scale. The camera performs a smooth, cinematic slow push-in, with natural parallax between the foreground dust, the astronaut, and the distant starfield. Ultra-realistic detail, physically accurate low-gravity motion, cinematic lighting, and a breath-taking, movie-like shot.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static.",
    )

    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--frame_rate", type=float, default=25.0)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--cpu_offload", action="store_true")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="samples",
        help="Output directory for generated video",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="ltx2_sample_video.mp4",
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

    video, audio = pipeline(
        image=Image.open(args.image_path),
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device=args.device).manual_seed(args.seed),
        output_type="np",
        return_dict=False,
    )

    # Convert video to uint8 (but keep as NumPy array)
    video = (video * 255).round().astype("uint8")
    video = torch.from_numpy(video)

    encode_video(
        video[0],
        fps=args.frame_rate,
        audio=audio[0].float().cpu(),
        audio_sample_rate=pipeline.vocoder.config.output_sampling_rate,  # should be 24000
        output_path=os.path.join(args.output_dir, args.output_filename),
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
