# Models

🤗 Diffusers provides pretrained models for popular algorithms and modules to create custom diffusion systems. The primary function of models is to denoise an input sample as modeled by the distribution \\(p_{\theta}(x_{t-1}|x_{t})\\).

All models are built from the base [`ModelMixin`] class which is a [`torch.nn.module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) providing basic functionality for saving and loading models, locally and from the Hugging Face Hub.

## ModelMixin
[[autodoc]] ModelMixin

## FlaxModelMixin

[[autodoc]] FlaxModelMixin

## PushToHubMixin

[[autodoc]] utils.PushToHubMixin