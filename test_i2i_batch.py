import torch
from diffusers.pipelines.glm_image import GlmImagePipeline
from PIL import Image

pipe = GlmImagePipeline.from_pretrained("/workspace/GLM-Image", torch_dtype=torch.bfloat16, device_map="cuda")

# Load condition image(s)
# For testing, create a simple test image or load from file
def create_test_image(size=(512, 512), color=(128, 128, 128)):
    """Create a simple test image"""
    return Image.new("RGB", size, color)

# You can replace this with actual images:
# condition_image = Image.open("your_condition_image.png")
condition_image = create_test_image(size=(512, 512), color=(100, 150, 200))

# =============================================================================
# Test 1: batch=1 (single prompt, single condition image)
# =============================================================================
print("=" * 50)
print("Test 1: batch=1 (single prompt)")
print("=" * 50)

prompt_single = "A beautiful landscape painting in the style of this image"

images_single = pipe(
    prompt=prompt_single,
    image=[condition_image],  # i2i requires image parameter
    height=32 * 32,
    width=36 * 32,
    num_inference_steps=50,
    guidance_scale=1.5,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images

for i, img in enumerate(images_single):
    img.save(f"i2i_batch1_output_{i}.png")
    print(f"Saved i2i_batch1_output_{i}.png")

# =============================================================================
# Test 2: batch>1 (multiple prompts, shared condition images)
# Note: Currently all prompts share the same condition images
# =============================================================================
print("\n" + "=" * 50)
print("Test 2: batch>1 (multiple prompts, shared condition images)")
print("=" * 50)

prompts_batch = [
    "A beautiful landscape painting in the style of this image",
    "A surreal dreamscape inspired by this image",
]

images_batch = pipe(
    prompt=prompts_batch,
    image=[condition_image],  # Shared across all prompts
    height=32 * 32,
    width=36 * 32,
    num_inference_steps=50,
    guidance_scale=1.5,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images

for i, img in enumerate(images_batch):
    img.save(f"i2i_batch2_output_{i}.png")
    print(f"Saved i2i_batch2_output_{i}.png")

# =============================================================================
# Test 3: batch>1 with num_images_per_prompt
# =============================================================================
print("\n" + "=" * 50)
print("Test 3: batch>1 with num_images_per_prompt=2")
print("=" * 50)

images_multi = pipe(
    prompt=prompts_batch,
    image=[condition_image],
    height=32 * 32,
    width=36 * 32,
    num_inference_steps=50,
    guidance_scale=1.5,
    num_images_per_prompt=2,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images

for i, img in enumerate(images_multi):
    img.save(f"i2i_batch3_output_{i}.png")
    print(f"Saved i2i_batch3_output_{i}.png")

# =============================================================================
# Test 4: Multiple condition images
# =============================================================================
print("\n" + "=" * 50)
print("Test 4: Multiple condition images")
print("=" * 50)

condition_image_2 = create_test_image(size=(512, 512), color=(200, 100, 100))

images_multi_cond = pipe(
    prompt="A creative artwork combining elements from these images",
    image=[condition_image, condition_image_2],  # Multiple condition images
    height=32 * 32,
    width=36 * 32,
    num_inference_steps=50,
    guidance_scale=1.5,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images

for i, img in enumerate(images_multi_cond):
    img.save(f"i2i_batch4_output_{i}.png")
    print(f"Saved i2i_batch4_output_{i}.png")

print("\n" + "=" * 50)
print("All tests completed!")
print("=" * 50)
