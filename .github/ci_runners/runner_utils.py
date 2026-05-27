import json
import time
from contextlib import contextmanager

import torch
from PIL import Image
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def detect_device() -> tuple[str, torch.dtype]:
    try:
        import torch_npu  # noqa: F401
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu", torch.bfloat16
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


@contextmanager
def timer():
    start = time.perf_counter()
    elapsed = 0.0
    try:
        yield lambda: elapsed
    finally:
        elapsed = time.perf_counter() - start


def validate_image(image: Image.Image, expected_width: int, expected_height: int) -> dict:
    w, h = image.size
    dimensions_ok = (w == expected_width and h == expected_height)

    arr = np.array(image.convert("RGB"), dtype=np.float32)
    non_black = bool(arr.max() > 5.0)

    return {"dimensions_ok": dimensions_ok, "non_black": non_black}


def compare_with_reference(image: Image.Image, ref_path: str) -> dict:
    if not HAS_SKIMAGE:
        return {"error": "skimage not installed, cannot compute PSNR/SSIM"}
    try:
        ref = Image.open(ref_path).convert("RGB")
    except FileNotFoundError:
        return {"error": f"reference image not found: {ref_path}"}

    img_arr = np.array(image, dtype=np.float64)
    ref_arr = np.array(ref, dtype=np.float64)

    if img_arr.shape != ref_arr.shape:
        return {"error": f"shape mismatch: {img_arr.shape} vs {ref_arr.shape}"}

    mse = np.mean((img_arr - ref_arr) ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))

    ssim_val = ssim(ref_arr, img_arr, channel_axis=-1, data_range=255)

    return {"psnr": round(psnr, 2), "ssim": round(ssim_val, 4)}


def to_uint8_hwc(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0 and arr.min() >= 0.0:
                arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    return arr


def frame_to_pil(frame):
    if isinstance(frame, Image.Image):
        return frame
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if isinstance(frame, np.ndarray):
        frame = to_uint8_hwc(frame)
        return Image.fromarray(frame)
    raise TypeError(f"Unsupported frame type: {type(frame)}")


def extract_frames_as_pil(result_obj) -> list:
    frames_data = result_obj.frames

    if isinstance(frames_data, torch.Tensor):
        frames_data = frames_data.detach().cpu().numpy()

    if isinstance(frames_data, np.ndarray):
        if frames_data.ndim == 5:
            video = frames_data[0]
        elif frames_data.ndim == 4:
            video = frames_data
        else:
            raise ValueError(f"Unexpected ndarray shape for frames: {frames_data.shape}")
        return [frame_to_pil(f) for f in video]

    if isinstance(frames_data, (list, tuple)):
        if len(frames_data) == 0:
            raise ValueError("result.frames is empty")
        first = frames_data[0]
        if isinstance(first, (list, tuple, np.ndarray, torch.Tensor, Image.Image)):
            video = first
        else:
            video = frames_data
        if isinstance(video, torch.Tensor):
            video = video.detach().cpu().numpy()
        if isinstance(video, np.ndarray):
            if video.ndim != 4:
                raise ValueError(f"Unexpected video ndarray shape: {video.shape}")
            return [frame_to_pil(f) for f in video]
        if isinstance(video, (list, tuple)):
            return [frame_to_pil(f) for f in video]

    raise TypeError(f"Unsupported result.frames type: {type(frames_data)}")


def apply_tp2(pipe):
    """Split transformer blocks across 2 NPUs for A14B models."""
    for tf_name in ("transformer", "transformer_2"):
        if not hasattr(pipe, tf_name) or getattr(pipe, tf_name) is None:
            continue
        t = getattr(pipe, tf_name)
        blocks = t.blocks
        total = len(blocks)
        half = total // 2

        for name, mod in t.named_children():
            if name == "blocks":
                continue
            mod.to("npu:0")

        first_half = blocks[:half]
        second_half = blocks[half:]
        for blk in first_half:
            blk.to("npu:0")
        for blk in second_half:
            blk.to("npu:1")

        class DeviceAwareBlock(torch.nn.Module):
            def __init__(self, block):
                super().__init__()
                self.block = block
            def forward(self, hidden_states, encoder_hidden_states, timestep, rotary_emb):
                target_device = next(self.block.parameters()).device
                if hidden_states.device != target_device:
                    hidden_states = hidden_states.to(target_device)
                if encoder_hidden_states.device != target_device:
                    encoder_hidden_states = encoder_hidden_states.to(target_device)
                return self.block(hidden_states, encoder_hidden_states, timestep, rotary_emb)

        new_blocks = [DeviceAwareBlock(blk) for blk in t.blocks]
        t.blocks = torch.nn.ModuleList(new_blocks)

        _orig_proj_out = t.proj_out
        class DeviceAwareProjOut(torch.nn.Module):
            def __init__(self, proj):
                super().__init__()
                self.proj = proj
            def forward(self, x):
                if x.device != self.proj.weight.device:
                    x = x.to(self.proj.weight.device)
                return self.proj(x)
        t.proj_out = DeviceAwareProjOut(_orig_proj_out)

    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae = pipe.vae.to("npu:0")
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        pipe.text_encoder = pipe.text_encoder.to("npu:0")

    pipe.__class__ = type(
        pipe.__class__.__name__,
        (pipe.__class__,),
        {"_execution_device": property(lambda self: torch.device("npu:0"))}
    )
