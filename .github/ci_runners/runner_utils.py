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
