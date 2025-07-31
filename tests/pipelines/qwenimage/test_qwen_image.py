import torch

from diffusers import (
    QwenImagePipeline,
)


if __name__ == "__main__":
    import os

    pipeline = QwenImagePipeline.from_pretrained("/cpfs03/user/jingqi/qwen-image-release-0728")
    print("pipeline loaded")
    # import ipdb; ipdb.set_trace()
    pipeline.to(torch.bfloat16)
    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=None)
    # prompt = "A painting of a squirrel eating a burger"
    prompt = "现实主义风格的人像摄影作品，画面主体是一位容貌惊艳的女性面部特写。她拥有一头自然微卷的短发，发丝根根分明，蓬松的刘海修饰着额头，增添俏皮感。头上佩戴一顶绿色格子蕾丝边头巾，增添复古与柔美气息。身着一件简约绿色背心裙，在纯白色背景下格外突出。两只手分别握着半个红色桃子，双手轻轻贴在脸颊两侧，营造出可爱又富有创意的视觉效果。  人物表情生动，一只眼睛睁开，另一只微微闭合，展现出调皮与自信的神态。整体构图采用个性视角、非对称构图，聚焦人物主体，增强现场感和既视感。背景虚化处理，层次丰富，景深效果强烈，营造出低光氛围下浓厚的情绪张力。  画面细节精致，色彩生动饱满却不失柔和，呈现出富士胶片独有的温润质感。光影运用充满美学张力，带有轻微超现实的光效处理，提升整体画面高级感。整体风格为现实主义人像摄影，强调细腻的纹理与艺术化的光线表现，堪称一幅细节丰富、氛围拉满的杰作。超清，4K，电影级构图"
    inputs = {
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "height": 1328,
        "width": 1328,
        "num_inference_steps": 50,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }
    with torch.inference_mode():
        output = pipeline(**inputs)
        # import ipdb; ipdb.set_trace()
        output_image = output.images[0]
        output_image.save("output_image.png")
        print("image saved at", os.path.abspath("output_image.png"))
