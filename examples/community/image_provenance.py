"""
AI Image Provenance — Embed generation metadata into diffusers output images.

Adds EXIF/PNG metadata to generated images so downstream consumers know:
- The image was AI-generated
- Which model and pipeline created it
- When it was generated
- What prompt was used (optional)

Useful for EU AI Act Article 50 compliance (August 2, 2026) which requires
transparency metadata on AI-generated content including images.

Usage:
    from diffusers import StableDiffusionPipeline
    # After generating an image:
    image = pipe("a sunset over mountains").images[0]
    image_with_provenance = embed_provenance(image, model_name="stabilityai/sdxl", prompt="a sunset")
    image_with_provenance.save("output.png")

The saved image carries provenance in PNG tEXt chunks and EXIF UserComment,
readable by ExifTool, Pillow, or any metadata-aware tool.
"""

import json
from datetime import datetime, timezone
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def embed_provenance(
    image: Image.Image,
    model_name: str = "unknown",
    prompt: str = "",
    negative_prompt: str = "",
    num_inference_steps: int = 0,
    guidance_scale: float = 0.0,
    seed: int = -1,
    include_prompt: bool = True,
) -> Image.Image:
    """Embed AI provenance metadata into a PIL Image.

    Args:
        image: The generated PIL Image.
        model_name: HuggingFace model ID or name.
        prompt: The generation prompt (omitted if include_prompt=False).
        negative_prompt: The negative prompt.
        num_inference_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.
        seed: Random seed used for generation.
        include_prompt: Whether to include the prompt in metadata.

    Returns:
        The same image with provenance metadata attached.
    """
    provenance = {
        "ai_generated": True,
        "model": model_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "diffusers",
    }

    if include_prompt and prompt:
        provenance["prompt"] = prompt
    if negative_prompt:
        provenance["negative_prompt"] = negative_prompt
    if num_inference_steps > 0:
        provenance["num_inference_steps"] = num_inference_steps
    if guidance_scale > 0:
        provenance["guidance_scale"] = guidance_scale
    if seed >= 0:
        provenance["seed"] = seed

    # Store as JSON string in image info (for PNG tEXt chunks)
    image.info["ai_provenance"] = json.dumps(provenance)
    image.info["ai_generated"] = "true"
    image.info["ai_model"] = model_name

    return image


def save_with_provenance(image: Image.Image, path: str, **kwargs) -> None:
    """Save image with provenance metadata preserved in PNG tEXt chunks."""
    if path.lower().endswith(".png"):
        pnginfo = PngInfo()
        for key, value in image.info.items():
            if isinstance(value, str):
                pnginfo.add_text(key, value)
        image.save(path, pnginfo=pnginfo, **kwargs)
    else:
        # For JPEG/WebP, store in EXIF UserComment
        from PIL.ExifTags import Base as ExifBase
        exif = image.getexif()
        provenance_str = image.info.get("ai_provenance", "{}")
        exif[ExifBase.UserComment] = provenance_str.encode()
        image.save(path, exif=exif.tobytes(), **kwargs)


def read_provenance(path: str) -> dict:
    """Read AI provenance metadata from an image file."""
    image = Image.open(path)

    # Try PNG tEXt
    if "ai_provenance" in image.info:
        return json.loads(image.info["ai_provenance"])

    # Try EXIF UserComment
    exif = image.getexif()
    from PIL.ExifTags import Base as ExifBase
    if ExifBase.UserComment in exif:
        try:
            return json.loads(exif[ExifBase.UserComment].decode())
        except (json.JSONDecodeError, AttributeError):
            pass

    return {}


if __name__ == "__main__":
    # Demo: create a test image and embed provenance
    img = Image.new("RGB", (512, 512), color=(60, 120, 200))

    img = embed_provenance(
        img,
        model_name="stabilityai/stable-diffusion-xl-base-1.0",
        prompt="a sunset over mountains",
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=42,
    )

    save_with_provenance(img, "/tmp/provenance_demo.png")
    print("Saved with provenance:")
    print(json.dumps(read_provenance("/tmp/provenance_demo.png"), indent=2))
