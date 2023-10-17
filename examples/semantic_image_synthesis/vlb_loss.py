import torch
import torch.nn as nn


class VLBLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,p_mean:torch.Tensor,p_log_var:torch.Tensor,q_mean:torch.Tensor,q_log_var:torch.Tensor):
        return 0.5 * (
            -1.0
            + q_log_var
            - p_log_var
            + torch.exp(p_log_var - q_log_var)
            + ((p_mean - q_mean) ** 2) * torch.exp(-q_log_var)
        )