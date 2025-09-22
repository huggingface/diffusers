import sys
sys.path.append('/home/lyc/diffusers/src')  # 替换为你的 diffusers 路径

import torch
import logging

# 先配置 logging（在导入 diffusers 前）
logger = logging.getLogger()  # 根 logger
logger.setLevel(logging.DEBUG)

# 
output_path = '/home/lyc/diffusers_output/'

# 文件 handler
file_handler = logging.FileHandler(output_path + 'cogvideox.log', mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

import time
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from torch.nn import CosineSimilarity
from torch.nn.functional import cosine_similarity


# 全局变量：统一管理所有模块类型（attn, ff）
module_times = {'attn': {}, 'ff': {}}  # {type: {name: time}}
start_events = {'attn': {}, 'ff': {}}  # {type: {name: event}}

def pre_timing_hook(module_type, module, input, name=""):
    if name not in start_events[module_type]:
        start_events[module_type][name] = torch.cuda.Event(enable_timing=True)
    start_events[module_type][name].record()  # 记录开始事件

def post_timing_hook(module_type, module, input, output, name=""):
    end_event = torch.cuda.Event(enable_timing=True)
    end_event.record()
    end_event.synchronize()  # 只在结束时同步
    elapsed = start_events[module_type][name].elapsed_time(end_event)  # GPU 时间（ms）
    module_times[module_type][name] = module_times[module_type].get(name, 0) + elapsed
    # 无需在这里记录新 start_event（移除重复逻辑）

def compute_cosine_similarity(block_idx1, block_idx2, blocks, layer_name="to_k"):
    """计算两个块中指定层（to_k 或 to_v）的余弦相似度"""
    attn1 = blocks[block_idx1].attn1
    attn2 = blocks[block_idx2].attn1
    if layer_name == "to_k":
        weight1 = attn1.to_k.weight.data.flatten()
        weight2 = attn2.to_k.weight.data.flatten()
    elif layer_name == "to_v":
        weight1 = attn1.to_v.weight.data.flatten()
        weight2 = attn2.to_v.weight.data.flatten()
    else:
        raise ValueError("layer_name must be 'to_k' or 'to_v'")
    return cosine_similarity(weight1.unsqueeze(0), weight2.unsqueeze(0)).item()

def main():
    logger.info(logging.getLogger('diffusers.pipelines.cogvideox.pipeline_cogvideox').handlers)
    logger.info("Loading CogVideoXPipeline...")
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",  # 或CogVideoX-5b（需更多显存）
        torch_dtype=torch.float16  # 节省内存
    )

    # 优化：CPU卸载，适合低显存设备
    pipe.enable_model_cpu_offload()
    logger.info("Model loaded and offloaded to CPU.")

    # 获取 Transformer 模块
    model = pipe.transformer

    # 访问Transformer块（假设模型有.model.transformer.blocks）
    blocks = model.transformer_blocks  # 列表 of TransformerBlock
    block_idx1, block_idx2 = 0, 1  # 比较第0和第1块，可改

    # 获取注意力层（CogVideoX用UNet-like Transformer，attn有to_q/to_k/to_v）
    attn11 = blocks[block_idx1].attn1
    attn12 = blocks[block_idx2].attn1

    # 提取to_k权重（.data避免梯度）
    k_weight1 = attn11.to_k.weight.data.flatten()  # flatten成1D向量
    k_weight2 = attn12.to_k.weight.data.flatten()

    # 计算余弦相似度（unsqueeze为batch dim）
    sim_k = cosine_similarity(k_weight1.unsqueeze(0), k_weight2.unsqueeze(0)).item()

    # 同理to_v
    v_weight1 = attn11.to_v.weight.data.flatten()
    v_weight2 = attn12.to_v.weight.data.flatten()
    sim_v = cosine_similarity(v_weight1.unsqueeze(0), v_weight2.unsqueeze(0)).item()

    print(f"Block {block_idx1} vs {block_idx2}:")
    print(f"to_k 余弦相似度: {sim_k:.6f}")
    print(f"to_v 余弦相似度: {sim_v:.6f}")
    print(f"两个block的余弦相似度: {compute_cosine_similarity(block_idx1, block_idx2, blocks, "to_k")}")

    # 注册 Hook：避免重叠，只匹配顶层模块（调整匹配条件）
    for name, module in model.named_modules():
        if "attn" in name.lower() and not any(sub in name for sub in ['proj', 'qkv']):  # 避免内部子模块重叠
            module.register_forward_pre_hook(lambda m, i, n=name: pre_timing_hook('attn', m, i, n))
            module.register_forward_hook(lambda m, i, o, n=name: post_timing_hook('attn', m, i, o, n))
            logger.info(f"注册 Hook 到 attn 模块: {name}")
        if "ff" in name.lower() and not any(sub in name for sub in ['net', 'proj']):  # 类似，避免 FF 子层
            module.register_forward_pre_hook(lambda m, i, n=name: pre_timing_hook('ff', m, i, n))
            module.register_forward_hook(lambda m, i, o, n=name: post_timing_hook('ff', m, i, o, n))
            logger.info(f"注册 Hook 到 ff 模块: {name}")

    # 输入提示
    prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
    logger.info(f"Using prompt: {prompt}")

    try:
        # 生成视频
        logger.info("Starting video generation...")
        video = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=10,  # 生成步骤越多，质量越高，但速度越慢
            height=480,  # 视频高度
            width=720,   # 视频宽度
            num_frames=9,  # 视频帧数, 8N+1, N<=6
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]
        logger.info(f"Generated video with {len(video)} frames, shape: {video[0].size}")

        # 导出视频
        export_to_video(video, output_path + "output_video.mp4", fps=16)
        logger.info("Video saved to output_video.mp4")

    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise

    finally:
        # 强制刷新日志，确保写入文件
        for handler in logger.handlers:
            handler.flush()

    # 输出总耗时
    total_attention_time = sum(module_times['attn'].values())
    print(f"所有 Attention 模块总耗时: {total_attention_time:.2f} ms")
    total_ff_time = sum(module_times['ff'].values())
    print(f"所有 FF 模块总耗时: {total_ff_time:.2f} ms")

if __name__ == "__main__":
    main()