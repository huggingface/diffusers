import sys
import os
sys.path.append('/home/lyc/diffusers/src')
# import torch
import logging

# 先配置 logging（在导入 diffusers 前）
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

output_path = '/home/lyc/diffusers_output/'
save_dir = '/home/lyc/diffusers_output/attn_maps'

# 文件 handler
file_handler = logging.FileHandler(output_path + 'cogvideox.log', mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import torch
import os, re, math, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers.hooks import ModelHook, HookRegistry
from diffusers.maputils import compute_query_redundancy_cosine_lowmem
from diffusers.maputils import save_redundancy_heatmap_lowmem
from matplotlib.colors import LinearSegmentedColormap
import torch.nn.functional as F
from diffusers.pipelines import CogVideoXPipeline
from diffusers.vishook import AttnCaptureHook, LatentFrameVizHook, TransformerStepHook, assign_layer_ids_and_register

# 通过，解耦：保存路径
def testAttnCaptureHook(pipe):
    shared_state = {"last_timestep": None, "step_index": -1, "timestep": None}
    HookRegistry.check_if_exists_or_initialize(pipe.transformer).register_hook(
        TransformerStepHook(shared_state), "step_hook"
    )
    # 配置：只捕获5层，第10步，立即处理，记得设置计算的是哪种图，因为都一起计算的话显存不够？
    attn_hook = AttnCaptureHook(
        shared_state,
        target_layers=list(range(3)),  # 前5层
        target_heads=[0, 1],            # 前2个头
        target_steps=[10],              # 只捕获第10步
        store_limit_per_layer=1,
        force_square=True,
        max_sequence_length=51200,
        process_immediately=True,       # 立即处理
        attn_qk_map=True,          # 计算QK的attention score map
        redundancy_q_map=False,   # 计算query之间的冗余度
        output_dir=save_dir,            # 输出目录
    )
    total_layers = assign_layer_ids_and_register(pipe.transformer, attn_hook)
    logger.info(f"Registered attention capture on {total_layers} layers.")

def testLatentFrameVizHook(pipe):
    # 2) 注册 hooks
    shared_state = {"last_timestep": None, "step_index": -1, "timestep": None}
    HookRegistry.check_if_exists_or_initialize(pipe.transformer).register_hook(
        TransformerStepHook(shared_state), "step_hook"
    )

    # 2).a 注册帧级 latent 可视化 hook，只在第10步、第 {3,12} 帧做可视化，最多 4096 tokens：
    frame_viz = LatentFrameVizHook(
        save_root="/home/lyc/diffusers_output/frame_viz",
        target_steps=[12],
        target_frames=[4],  # 只看第12帧，注意，这里是被 vae 时间压缩过的。。。
        query_indices=[0, 128, 512],  # 在 Full Attention 图里额外展示这几个 query
        max_hw_tokens=9182,
        row_chunk=28,                 # 视内存调小
        cosine_device="cpu",
        cosine_dtype=torch.float32,
        temperature=1.0,
        decode_latents=True,          # 解码 latent
    )
    HookRegistry.check_if_exists_or_initialize(pipe.transformer).register_hook(frame_viz, "latent_frame_viz")



def main():
    print(torch.cuda.is_available())
    print(torch.version.cuda)

    # 1) 加载 pipeline
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()
    logger.info("Model loaded and offloaded to CPU.")

    # 2) 注册 hooks

    # 2).a 注册帧级 latent 可视化 hook
    testLatentFrameVizHook(pipe)

    # 2).b 注册 attention capture hook
    # testAttnCaptureHook(pipe)

    # 3) 推理
    prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest..."
    logger.info(f"Using prompt: {prompt}")

    logger.info("Starting video generation...")
    result = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=20,  # 增加步数以确保能捕获到第10步
        height=480,
        width=720,
        num_frames=17,
        guidance_scale=6,
        generator=torch.Generator(device="cuda").manual_seed(42) if torch.cuda.is_available() else None,
    )
    video = result.frames[0]
    logger.info(f"Generated video with {len(video)} frames, shape: {video[0].size}")

    # 4) 保存视频
    os.makedirs(output_path, exist_ok=True)
    export_to_video(video, os.path.join(output_path, "output_video.mp4"), fps=16)
    logger.info("Video saved to output_video.mp4")
    logger.info("All processing completed!")

if __name__ == "__main__":
    main()