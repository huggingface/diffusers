import numpy as np

import oneflow as flow
import torch

from transformers import OneFlowCLIPTextModel, CLIPTextModel

pretrained_model_path = "/home/ldp/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-4/snapshots/a304b1ab1b59dd6c3ba9c40705c29c6de4144096/text_encoder"

of_type = flow.float32
torch_type = torch.float32
of_type = flow.float16
torch_type = torch.float16

loading_kwargs = {'torch_dtype': of_type}
of_text_encoder = OneFlowCLIPTextModel.from_pretrained(pretrained_model_path, **loading_kwargs)
of_text_encoder = of_text_encoder.to("cuda")
loading_kwargs = {'torch_dtype': torch_type}
torch_text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, **loading_kwargs)
torch_text_encoder = torch_text_encoder.to("cuda")

text_ids = np.random.randint(1, 49407, (1, 77), np.int64)
of_input = flow.tensor(text_ids, device="cuda")
torch_input = torch.tensor(text_ids, device="cuda")

with flow.no_grad():
    of_text_embeddings = of_text_encoder(of_input, attention_mask=None)
    of_text_embeddings = of_text_embeddings[0]

with torch.no_grad():
    torch_text_embeddings = torch_text_encoder(torch_input, attention_mask=None)
    torch_text_embeddings = torch_text_embeddings[0]

out_1 = of_text_embeddings.cpu().numpy()
out_2 = torch_text_embeddings.cpu().numpy()
out_1 = out_1[~np.isnan(out_1)]
out_2 = out_2[~np.isnan(out_2)]
max_diff = np.amax(np.abs(out_1 - out_2))
print(f"max diff: {max_diff}")
mean_diff = np.mean(np.abs(out_1 - out_2))
print(f"mean diff: {mean_diff}")
