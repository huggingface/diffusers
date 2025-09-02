import torch
import logging
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# 设置日志
# 设置日志，保存到文件

# 设置日志，保存到文件
logging.basicConfig(
    level=logging.DEBUG,
    filename='cogvideox.log',  # 日志保存到当前目录的 cogvideox.log
    filemode='w',  # 'w' 覆盖旧日志，'a' 追加
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
)
logger = logging.getLogger(__name__)

def main():

    # 加载CogVideoX模型
    logger.info("Loading CogVideoXPipeline...")
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",  # 或CogVideoX-5b（需更多显存）
        torch_dtype=torch.bfloat16  # 节省内存
        # torch_dtype = torch.float16
    )

    # 优化：CPU卸载，适合低显存设备
    pipe.enable_model_cpu_offload()
    logger.info("Model loaded and offloaded to CPU.")

    # 输入提示
    prompt = "A cat wearing a hat dancing in a colorful garden."
    logger.info(f"Using prompt: {prompt}")

    try:
        # 生成视频
        logger.info("Starting video generation...")
        video = pipe(
            prompt=prompt,
            num_inference_steps=10,  # 调试时用小值
            height=480,
            width=720,
            guidance_scale=6.0,
            num_frames=5  # 调试时减少帧数
        ).frames[0]
        logger.info(f"Generated video with {len(video)} frames, shape: {video[0].shape}")

        # 导出视频
        export_to_video(video, "output_video.mp4", fps=8)
        logger.info("Video saved to output_video.mp4")

    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise


if __name__ == "__main__":
    main()