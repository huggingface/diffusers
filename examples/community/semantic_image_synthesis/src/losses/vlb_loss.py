import torch
import torch.nn as nn

class VLBLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,p_mean:torch.Tensor,p_log_var:torch.Tensor,q_mean:torch.Tensor,q_log_var:torch.Tensor):
        """KL Divergence from log_var
        We take the equation here (based on sigma = sqrt(log))

        log(x^1/2) = 1/2 . log(x)
        
        Args:
            p_mean (torch.Tensor): _description_
            p_log_var (torch.Tensor): _description_
            q_mean (torch.Tensor): _description_
            q_log_var (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        kld = -0.5 + 0.5*(q_log_var - p_log_var) + 0.5*(
            torch.exp(p_log_var)/torch.exp(q_log_var) 
            + 
            (p_mean - q_mean)**2/torch.exp(q_log_var))
        return kld