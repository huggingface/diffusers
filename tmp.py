import torch
from diffusers import SanaPipeline, SanaVideoPipeline, UniPCMultistepScheduler
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video


def sana_video():

    model_id = "Efficient-Large-Model/sana_video_v2_diffusers"
    pipe = SanaVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
    pipe.vae.to(torch.float32)
    pipe.text_encoder.to(torch.bfloat16)
    pipe.to("cuda")

    prompt = "Extreme close-up of a thoughtful, gray-haired professor in his 60s, sitting motionless in a Paris café, dressed in a wool coat and beret, pondering the universe. His subtle closed-mouth smile reveals an answer. Golden light, cinematic depth of field, Paris streets blurred in the background. Cinematic 35mm film."
    negative_prompt = "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience."

    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=480,
        width=832,
        frames=81,
        guidance_scale=6,
        num_inference_steps=50,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]

    export_to_video(video, "sana_v2.mp4", fps=16)


def profile_sana_video():
    from tqdm import tqdm
    import time
    model_id = "sana_video"
    pipe = SanaVideoPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
    pipe.vae.to(torch.float32)
    pipe.text_encoder.to(torch.bfloat16)
    pipe.to("cuda")

    prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    for i in tqdm(range(1), desc="Warmup"):
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=480,
            width=832,
            frames=81,
            guidance_scale=6,
            num_inference_steps=30,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]

    n = 1
    time_start = time.time()
    for i in tqdm(range(n), desc=f"Inference {n} times"):
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=480,
            width=832,
            frames=81,
            guidance_scale=6,
            num_inference_steps=50,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]

    time_end = time.time()
    print(f"Time taken: {(time_end - time_start)/n} seconds/video, {n / (time_end - time_start) * 81} fps")


def wan():
    import torch
    from diffusers.utils import export_to_video
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    # model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
    pipe.to("cuda")

    prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=720,
        width=1280,
        num_frames=81,
        guidance_scale=5.0,
    ).frames[0]
    export_to_video(output, "output.mp4", fps=16)


if __name__ == "__main__":
    sana_video()
    # profile_sana_video()
    # wan()
