import torch


class GGUFParameter(torch.nn.Parameter):
    def __init__(self, data):
        super().__init__()
