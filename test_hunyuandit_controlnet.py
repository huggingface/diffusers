# from diffusers import HunyuanDiT2DControlNetModel, HunyuanDiTControlNetPipeline
# import torch
# controlnet = HunyuanDiT2DControlNetModel.from_pretrained("../HunyuanDiT-v1.1-ControlNet-Pose", torch_dtype=torch.float16)

# pipe = HunyuanDiTControlNetPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers", controlnet=controlnet, torch_dtype=torch.float16)
# pipe.to("cuda")

# from diffusers.utils import load_image
# cond_image = load_image('/apdcephfs_cq8/share_1367250/xingchaoliu/HunyuanDiT/controlnet/asset/input/pose.jpg')
# cond_image.save('condition.png')

# #prompt="在茂密的森林中，一只黑白相间的熊猫静静地坐在绿树红花中，周围是山川和海洋。背景是白天的森林，光线充足"
# #prompt="在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围"
# prompt="一位亚洲女性，身穿绿色上衣，戴着紫色头巾和紫色围巾，站在黑板前。背景是黑板。照片采用近景、平视和居中构图的方式呈现真实摄影风格"
# #prompt="An Asian woman, dressed in a green top, wearing a purple headscarf and a purple scarf, stands in front of a blackboard. The background is the blackboard. The photo is presented in a close-up, eye-level, and centered composition, adopting a realistic photographic style"
# image = pipe(
#     prompt, #negative_prompt="错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺，",
#     height=1024,
#     width=1024,
#     control_image=cond_image,
#     num_inference_steps=50,
# ).images[0]
# image.save('img.png')

from diffusers import HunyuanDiT2DControlNetModel, HunyuanDiTControlNetPipeline
import torch
controlnet = HunyuanDiT2DControlNetModel.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Canny", torch_dtype=torch.float16)

pipe = HunyuanDiTControlNetPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers", controlnet=controlnet, torch_dtype=torch.float16)
pipe.to("cuda")

from diffusers.utils import load_image
cond_image = load_image('https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Canny/resolve/main/canny.jpg?download=true')

## You may also use English prompt as HunyuanDiT supports both English and Chinese
prompt="在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围"
prompt="At night, an ancient Chinese-style lion statue stands in front of the hotel, its eyes gleaming as if guarding the building. The background is the hotel entrance at night, with a close-up, eye-level, and centered composition. This photo presents a realistic photographic style, embodies Chinese sculpture culture, and reveals a mysterious atmosphere."
image = pipe(
    prompt,
    height=1024,
    width=1024,
    control_image=cond_image,
    num_inference_steps=50,
).images[0]