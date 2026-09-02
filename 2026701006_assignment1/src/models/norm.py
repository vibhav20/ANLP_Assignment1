"""
- LayerNorm -> C1, C2, C3, C5
- RMSNorm   ->  C4
"""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        """
        x: (..., d_model)
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class RMSNorm(nn.Module):
    """
    C4
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """
        x: (..., d_model)
        """
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.gamma * x_norm
