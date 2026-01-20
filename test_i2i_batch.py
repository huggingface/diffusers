import torch
from diffusers.pipelines.glm_image import GlmImagePipeline
from PIL import Image

pipe = GlmImagePipeline.from_pretrained("/workspace/GLM-Image", torch_dtype=torch.bfloat16, device_map="cuda")

# Load test images
condition_image_0 = Image.open("./output/output_batch_1.png")
condition_image_1 = Image.open("./output/output_batch_2.png")

# =============================================================================
# Test 1: batch=1, single condition image
# =============================================================================
print("=" * 50)
print("Test 1: batch=1, single condition image")
print("=" * 50)

images = pipe(
    prompt="Make the man raise his arm and open his mouth",
    image=[condition_image_0],
    height=32 * 32,
    width=36 * 32,
    num_inference_steps=50,
    guidance_scale=1.5,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images

for i, img in enumerate(images):
    img.save(f"i2i_test1_output_{i}.png")
    print(f"Saved i2i_test1_output_{i}.png")

# =============================================================================
# Test 2: batch=1, multiple condition images
# =============================================================================
print("\n" + "=" * 50)
print("Test 2: batch=1, multiple condition images")
print("=" * 50)

images = pipe(
    prompt="A creative artwork combining elements from these images",
    image=[condition_image_0, condition_image_1],
    height=32 * 32,
    width=36 * 32,
    num_inference_steps=50,
    guidance_scale=1.5,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images

for i, img in enumerate(images):
    img.save(f"i2i_test2_output_{i}.png")
    print(f"Saved i2i_test2_output_{i}.png")

# =============================================================================
# Test 3: batch>1, each prompt with 1 condition image (homogeneous)
# =============================================================================
print("\n" + "=" * 50)
print("Test 3: batch=2, each prompt with 1 condition image")
print("=" * 50)

images = pipe(
    prompt=[
        "Make the man raise his arm and open his mouth",
        "Make the man jump in the air",
    ],
    image=[
        [condition_image_0],  # 1 image for prompt 0
        [condition_image_1],  # 1 image for prompt 1
    ],
    height=32 * 32,
    width=36 * 32,
    num_inference_steps=50,
    guidance_scale=1.5,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images

for i, img in enumerate(images):
    img.save(f"i2i_test3_output_{i}.png")
    print(f"Saved i2i_test3_output_{i}.png")

# =============================================================================
# Test 4: batch>1, each prompt with 2 condition images (homogeneous)
# =============================================================================
print("\n" + "=" * 50)
print("Test 4: batch=2, each prompt with 2 condition images")
print("=" * 50)

images = pipe(
    prompt=[
        "Combine these two images into an artwork",
        "Merge these scenes creatively",
    ],
    image=[
        [condition_image_0, condition_image_1],  # 2 images for prompt 0
        [condition_image_1, condition_image_0],  # 2 images for prompt 1
    ],
    height=32 * 32,
    width=36 * 32,
    num_inference_steps=50,
    guidance_scale=1.5,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images

for i, img in enumerate(images):
    img.save(f"i2i_test4_output_{i}.png")
    print(f"Saved i2i_test4_output_{i}.png")

# =============================================================================
# Test 5: batch>1 with num_images_per_prompt > 1
# =============================================================================
print("\n" + "=" * 50)
print("Test 5: batch=2, 1 condition image each, num_images_per_prompt=2")
print("=" * 50)

images = pipe(
    prompt=[
        "Make the man raise his arm",
        "Make the man wave hello",
    ],
    image=[
        [condition_image_0],
        [condition_image_1],
    ],
    height=32 * 32,
    width=36 * 32,
    num_inference_steps=50,
    guidance_scale=1.5,
    num_images_per_prompt=2,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images

for i, img in enumerate(images):
    img.save(f"i2i_test5_output_{i}.png")
    print(f"Saved i2i_test5_output_{i}.png")

print("\n" + "=" * 50)
print("All tests completed!")
print("=" * 50)