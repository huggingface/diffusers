import torch
from diffusers import CogView3PlusTransformer2DModel

model = CogView3PlusTransformer2DModel.from_pretrained("/share/home/zyx/Models/CogView3Plus_hf/transformer",torch_dtype=torch.bfloat16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

batch_size = 1
hidden_states = torch.ones((batch_size, 16, 256, 256), device=device, dtype=torch.bfloat16)
timestep = torch.full((batch_size,), 999.0, device=device, dtype=torch.bfloat16)
y = torch.ones((batch_size, 1536), device=device, dtype=torch.bfloat16)

# 模拟调用 forward 方法
outputs = model(
    hidden_states=hidden_states,  # hidden_states 输入
    timestep=timestep,  # timestep 输入
    y=y,  # 标签输入
    block_controlnet_hidden_states=None,  # 如果不需要，可以忽略
    return_dict=True,  # 保持默认值
    target_size=[(2048, 2048)],
)

# 输出模型结果
print("Output shape:", outputs.sample.shape)