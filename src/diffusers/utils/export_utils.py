from __future__ import annotations

import io
import random
import struct
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from itertools import chain

import numpy as np
import PIL.Image
import PIL.ImageOps

from .import_utils import BACKENDS_MAPPING, is_imageio_available, is_opencv_available
from .logging import get_logger


global_rng = random.Random()

logger = get_logger(__name__)


@contextmanager
def buffered_writer(raw_f):
    f = io.BufferedWriter(raw_f)
    yield f
    f.flush()


def export_to_gif(image: list[PIL.Image.Image], output_gif_path: str = None, fps: int = 10) -> str:
    if output_gif_path is None:
        output_gif_path = tempfile.NamedTemporaryFile(suffix=".gif").name

    image[0].save(
        output_gif_path,
        save_all=True,
        append_images=image[1:],
        optimize=False,
        duration=1000 // fps,
        loop=0,
    )
    return output_gif_path


def export_to_ply(mesh, output_ply_path: str = None):
    """
    Write a PLY file for a mesh.
    """
    if output_ply_path is None:
        output_ply_path = tempfile.NamedTemporaryFile(suffix=".ply").name

    coords = mesh.verts.detach().cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    rgb = np.stack([mesh.vertex_channels[x].detach().cpu().numpy() for x in "RGB"], axis=1)

    with buffered_writer(open(output_ply_path, "wb")) as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(bytes(f"element vertex {len(coords)}\n", "ascii"))
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        if rgb is not None:
            f.write(b"property uchar red\n")
            f.write(b"property uchar green\n")
            f.write(b"property uchar blue\n")
        if faces is not None:
            f.write(bytes(f"element face {len(faces)}\n", "ascii"))
            f.write(b"property list uchar int vertex_index\n")
        f.write(b"end_header\n")

        if rgb is not None:
            rgb = (rgb * 255.499).round().astype(int)
            vertices = [
                (*coord, *rgb)
                for coord, rgb in zip(
                    coords.tolist(),
                    rgb.tolist(),
                )
            ]
            format = struct.Struct("<3f3B")
            for item in vertices:
                f.write(format.pack(*item))
        else:
            format = struct.Struct("<3f")
            for vertex in coords.tolist():
                f.write(format.pack(*vertex))

        if faces is not None:
            format = struct.Struct("<B3I")
            for tri in faces.tolist():
                f.write(format.pack(len(tri), *tri))

    return output_ply_path


def export_to_obj(mesh, output_obj_path: str = None):
    if output_obj_path is None:
        output_obj_path = tempfile.NamedTemporaryFile(suffix=".obj").name

    verts = mesh.verts.detach().cpu().numpy()
    faces = mesh.faces.cpu().numpy()

    vertex_colors = np.stack([mesh.vertex_channels[x].detach().cpu().numpy() for x in "RGB"], axis=1)
    vertices = [
        "{} {} {} {} {} {}".format(*coord, *color) for coord, color in zip(verts.tolist(), vertex_colors.tolist())
    ]

    faces = ["f {} {} {}".format(str(tri[0] + 1), str(tri[1] + 1), str(tri[2] + 1)) for tri in faces.tolist()]

    combined_data = ["v " + vertex for vertex in vertices] + faces

    with open(output_obj_path, "w") as f:
        f.writelines("\n".join(combined_data))


def _legacy_export_to_video(
    video_frames: list[np.ndarray] | list[PIL.Image.Image], output_video_path: str = None, fps: int = 10
):
    if is_opencv_available():
        import cv2
    else:
        raise ImportError(BACKENDS_MAPPING["opencv"][1].format("export_to_video"))
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, c = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)

    return output_video_path


def export_to_video(
    video_frames: list[np.ndarray] | list[PIL.Image.Image],
    output_video_path: str = None,
    fps: int = 10,
    quality: float = 5.0,
    bitrate: int | None = None,
    macro_block_size: int | None = 16,
) -> str:
    """
    quality:
        Video output quality. Default is 5. Uses variable bit rate. Highest quality is 10, lowest is 0. Set to None to
        prevent variable bitrate flags to FFMPEG so you can manually specify them using output_params instead.
        Specifying a fixed bitrate using `bitrate` disables this parameter.

    bitrate:
        Set a constant bitrate for the video encoding. Default is None causing `quality` parameter to be used instead.
        Better quality videos with smaller file sizes will result from using the `quality` variable bitrate parameter
        rather than specifying a fixed bitrate with this parameter.

    macro_block_size:
        Size constraint for video. Width and height, must be divisible by this number. If not divisible by this number
        imageio will tell ffmpeg to scale the image up to the next closest size divisible by this number. Most codecs
        are compatible with a macroblock size of 16 (default), some can go smaller (4, 8). To disable this automatic
        feature set it to None or 1, however be warned many players can't decode videos that are odd in size and some
        codecs will produce poor results or fail. See https://en.wikipedia.org/wiki/Macroblock.
    """
    # TODO: Dhruv. Remove by Diffusers release 0.33.0
    # Added to prevent breaking existing code
    if not is_imageio_available():
        logger.warning(
            (
                "It is recommended to use `export_to_video` with `imageio` and `imageio-ffmpeg` as a backend. \n"
                "These libraries are not present in your environment. Attempting to use legacy OpenCV backend to export video. \n"
                "Support for the OpenCV backend will be deprecated in a future Diffusers version"
            )
        )
        return _legacy_export_to_video(video_frames, output_video_path, fps)

    if is_imageio_available():
        import imageio
    else:
        raise ImportError(BACKENDS_MAPPING["imageio"][1].format("export_to_video"))

    try:
        imageio.plugins.ffmpeg.get_exe()
    except AttributeError:
        raise AttributeError(
            (
                "Found an existing imageio backend in your environment. Attempting to export video with imageio. \n"
                "Unable to find a compatible ffmpeg installation in your environment to use with imageio. Please install via `pip install imageio-ffmpeg"
            )
        )

    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    with imageio.get_writer(
        output_video_path, fps=fps, quality=quality, bitrate=bitrate, macro_block_size=macro_block_size
    ) as writer:
        for frame in video_frames:
            writer.append_data(frame)

    return output_video_path


def _import_av():
    try:
        import av
    except ImportError as error:
        raise ImportError(
            "PyAV is required to use `encode_video`. You can install it with `pip install av`."
        ) from error

    return av


def _prepare_audio_stream(container, audio_sample_rate: int):
    """
    Prepare the audio stream for writing.
    """
    audio_stream = container.add_stream("aac", rate=audio_sample_rate)
    audio_stream.codec_context.sample_rate = audio_sample_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, audio_sample_rate)
    return audio_stream


def _resample_audio(container, audio_stream, frame_in, av_module) -> None:
    cc = audio_stream.codec_context

    # Use the encoder's format/layout/rate as the target.
    target_format = cc.format or "fltp"  # AAC is usually fltp.
    target_layout = cc.layout or "stereo"
    target_rate = cc.sample_rate or frame_in.sample_rate

    audio_resampler = av_module.audio.resampler.AudioResampler(
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

    # Flush audio encoder.
    for packet in audio_stream.encode():
        container.mux(packet)


def _write_audio(
    container,
    audio_stream,
    samples: "torch.Tensor",
    audio_sample_rate: int,
    av_module,
) -> None:
    import torch

    if samples.ndim == 1:
        samples = samples[:, None]

    if samples.shape[1] != 2 and samples.shape[0] == 2:
        samples = samples.T

    if samples.shape[1] != 2:
        raise ValueError(f"Expected samples with 2 channels; got shape {samples.shape}.")

    # Convert to int16 packed for ingestion; resampler converts to the encoder format.
    if samples.dtype != torch.int16:
        samples = torch.clip(samples, -1.0, 1.0)
        samples = (samples * 32767.0).to(torch.int16)

    frame_in = av_module.AudioFrame.from_ndarray(
        samples.contiguous().reshape(1, -1).cpu().numpy(),
        format="s16",
        layout="stereo",
    )
    frame_in.sample_rate = audio_sample_rate

    _resample_audio(container, audio_stream, frame_in, av_module)


def encode_video(
    video: list[PIL.Image.Image] | np.ndarray | "torch.Tensor" | Iterator["torch.Tensor"],
    fps: int,
    output_path: str,
    audio: "torch.Tensor" | None = None,
    audio_sample_rate: int | None = None,
    video_chunks_number: int = 1,
) -> None:
    """
    Encodes a video with optional audio using the PyAV library. Based on code from the original LTX-2 repo:
    https://github.com/Lightricks/LTX-2/blob/4f410820b198e05074a1e92de793e3b59e9ab5a0/packages/ltx-pipelines/src/ltx_pipelines/utils/media_io.py#L182

    Args:
        video (`List[PIL.Image.Image]` or `np.ndarray` or `torch.Tensor`):
            A video tensor of shape [frames, height, width, channels] with integer pixel values in [0, 255]. If the
            input is a `np.ndarray`, it is expected to be a float array with values in [0, 1] (which is what pipelines
            usually return with `output_type="np"`).
        fps (`int`)
            The frames per second (FPS) of the encoded video.
        output_path (`str`):
            The path to save the encoded video to.
        audio (`torch.Tensor`, *optional*):
            An audio waveform of shape [audio_channels, samples].
        audio_sample_rate: (`int`, *optional*):
            The sampling rate of the audio waveform.
        video_chunks_number (`int`, *optional*, defaults to `1`):
            The number of chunks to split the video into for encoding. Each chunk will be encoded separately. The
            number of chunks to use often depends on the tiling config for the video VAE.
    """
    av = _import_av()
    import torch
    from tqdm import tqdm

    if isinstance(video, list) and isinstance(video[0], PIL.Image.Image):
        # Pipeline output_type="pil"; assumes each image is in "RGB" mode.
        video_frames = [np.array(frame) for frame in video]
        video = np.stack(video_frames, axis=0)
        video = torch.from_numpy(video)
    elif isinstance(video, np.ndarray):
        # Pipeline output_type="np".
        is_denormalized = np.logical_and(np.zeros_like(video) <= video, video <= np.ones_like(video))
        if np.all(is_denormalized):
            video = (video * 255).round().astype("uint8")
        else:
            logger.warning(
                "Supplied `numpy.ndarray` does not have values in [0, 1]. The values will be assumed to be pixel "
                "values in [0, ..., 255] and will be used as is."
            )
        video = torch.from_numpy(video)

    if isinstance(video, torch.Tensor):
        # Split into video_chunks_number along the frame dimension.
        video = torch.tensor_split(video, video_chunks_number, dim=0)
        video = iter(video)

    first_chunk = next(video)

    _, height, width, _ = first_chunk.shape

    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    if audio is not None:
        if audio_sample_rate is None:
            raise ValueError("audio_sample_rate is required when audio is provided")

        audio_stream = _prepare_audio_stream(container, audio_sample_rate)

    for video_chunk in tqdm(chain([first_chunk], video), total=video_chunks_number, desc="Encoding video chunks"):
        video_chunk_cpu = video_chunk.to("cpu").numpy()
        for frame_array in video_chunk_cpu:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

    # Flush video encoder.
    for packet in stream.encode():
        container.mux(packet)

    if audio is not None:
        _write_audio(container, audio_stream, audio, audio_sample_rate, av)

    container.close()
