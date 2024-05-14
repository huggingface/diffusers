import io
import random
import struct
import tempfile
from contextlib import contextmanager
from typing import List, Optional, Union

import numpy as np
import PIL.Image
import PIL.ImageOps
import torch

from .import_utils import (
    BACKENDS_MAPPING,
    is_matplotlib_available,
    is_opencv_available,
)
from .logging import get_logger


global_rng = random.Random()

logger = get_logger(__name__)


@contextmanager
def buffered_writer(raw_f):
    f = io.BufferedWriter(raw_f)
    yield f
    f.flush()


def export_to_gif(image: List[PIL.Image.Image], output_gif_path: str = None, fps: int = 10) -> str:
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


def export_to_video(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]], output_video_path: str = None, fps: int = 10
) -> str:
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


def colormap(
    image: Union[np.ndarray, torch.Tensor],
    cmap: str = "Spectral",
    bytes: bool = False,
    _force_method: Optional[str] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Converts a monochrome image into an RGB image by applying the specified colormap. This function mimics the behavior
    of matplotlib.colormaps, but allows the user to use the most discriminative color map "Spectral" without having to
    install or import matplotlib. For all other cases, the function will attempt to use the native implementation.

    Args:
        image: 2D tensor of values between 0 and 1, either as np.ndarray or torch.Tensor.
        cmap: Colormap name.
        bytes: Whether to return the output as uint8 or floating point image.
        _force_method:
            Can be used to specify whether to use the native implementation (`"matplotlib"`), the efficient custom
            implementation of the "Spectral" color map (`"custom"`), or rely on autodetection (`None`, default).

    Returns:
        An RGB-colorized tensor corresponding to the input image.
    """
    if not (torch.is_tensor(image) or isinstance(image, np.ndarray)):
        raise ValueError("Argument must be a numpy array or torch tensor.")
    if _force_method not in (None, "matplotlib", "custom"):
        raise ValueError("_force_method must be either `None`, `'matplotlib'` or `'custom'`.")

    def method_matplotlib(image, cmap, bytes=False):
        if is_matplotlib_available():
            import matplotlib
        else:
            return None

        arg_is_pt, device = torch.is_tensor(image), None
        if arg_is_pt:
            image, device = image.cpu().numpy(), image.device

        if cmap not in matplotlib.colormaps:
            raise ValueError(
                f"Unexpected color map {cmap}; available options are: {', '.join(list(matplotlib.colormaps.keys()))}"
            )

        cmap = matplotlib.colormaps[cmap]
        out = cmap(image, bytes=bytes)  # [?,4]
        out = out[..., :3]  # [?,3]

        if arg_is_pt:
            out = torch.tensor(out, device=device)

        return out

    def method_custom(image, cmap, bytes=False):
        arg_is_np = isinstance(image, np.ndarray)
        if arg_is_np:
            image = torch.tensor(image)
        if image.dtype == torch.uint8:
            image = image.float() / 255
        else:
            image = image.float()

        if cmap != "Spectral":
            raise ValueError("Only 'Spectral' color map is available without installing matplotlib.")

        _Spectral_data = (  # Taken from matplotlib/_cm.py
            (0.61960784313725492, 0.003921568627450980, 0.25882352941176473),  # 0.0 -> [0]
            (0.83529411764705885, 0.24313725490196078, 0.30980392156862746),
            (0.95686274509803926, 0.42745098039215684, 0.2627450980392157),
            (0.99215686274509807, 0.68235294117647061, 0.38039215686274508),
            (0.99607843137254903, 0.8784313725490196, 0.54509803921568623),
            (1.0, 1.0, 0.74901960784313726),
            (0.90196078431372551, 0.96078431372549022, 0.59607843137254901),
            (0.6705882352941176, 0.8666666666666667, 0.64313725490196083),
            (0.4, 0.76078431372549016, 0.6470588235294118),
            (0.19607843137254902, 0.53333333333333333, 0.74117647058823533),
            (0.36862745098039218, 0.30980392156862746, 0.63529411764705879),  # 1.0 -> [K-1]
        )

        cmap = torch.tensor(_Spectral_data, dtype=torch.float, device=image.device)  # [K,3]
        K = cmap.shape[0]

        pos = image.clamp(min=0, max=1) * (K - 1)
        left = pos.long()
        right = (left + 1).clamp(max=K - 1)

        d = (pos - left.float()).unsqueeze(-1)
        left_colors = cmap[left]
        right_colors = cmap[right]

        out = (1 - d) * left_colors + d * right_colors

        if bytes:
            out = (out * 255).to(torch.uint8)

        if arg_is_np:
            out = out.numpy()

        return out

    if _force_method is None and torch.is_tensor(image) and cmap == "Spectral":
        return method_custom(image, cmap, bytes)

    out = None
    if _force_method != "custom":
        out = method_matplotlib(image, cmap, bytes)

    if _force_method == "matplotlib" and out is None:
        raise ImportError("Make sure to install matplotlib if you want to use a color map other than 'Spectral'.")

    if out is None:
        out = method_custom(image, cmap, bytes)

    return out


def export_depth_to_png(depth):
    depth_u16 = (depth * (2**16 - 1)).astype(np.uint16)
    out = PIL.Image.fromarray(depth_u16, mode="I;16")
    return out


def visualize_depth(
    depth: torch.FloatTensor,
    color_map: str = "Spectral",
    vis_min: float = 0.0,
    vis_max: float = 1.0,
) -> PIL.Image.Image:
    if depth.dim() != 2 or not torch.is_floating_point(depth):
        raise ValueError("Input should be a 2-dimensional floating point tensor of shape [H,W].")
    if vis_max <= vis_min:
        raise ValueError(f"Invalid colorization range: [{vis_min}, {vis_max}].")

    if vis_min != 0.0 or vis_max != 1.0:
        depth = (depth - vis_min) / (vis_max - vis_min)

    visualization = colormap(depth, cmap=color_map, bytes=True)  # [H,W,3]
    visualization = PIL.Image.fromarray(visualization.cpu().numpy())

    return visualization


def visualize_normals(
    normals: torch.FloatTensor,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
) -> PIL.Image.Image:
    if normals.dim() != 3 or not torch.is_floating_point(normals) or normals.shape[2] != 3:
        raise ValueError("Input should be a 3-dimensional floating point tensor of shape [H,W,3].")

    visualization = normals
    if any((flip_x, flip_y, flip_z)):
        flip_vec = [
            (-1) ** flip_x,
            (-1) ** flip_y,
            (-1) ** flip_z,
        ]
        visualization *= torch.tensor(flip_vec, dtype=normals.dtype, device=normals.device)

    visualization = (visualization + 1.0) * 0.5
    visualization = (visualization * 255).to(dtype=torch.uint8, device="cpu").numpy()
    visualization = PIL.Image.fromarray(visualization)

    return visualization
