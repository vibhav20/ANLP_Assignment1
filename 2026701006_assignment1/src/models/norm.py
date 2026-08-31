"""
norm.py

From-scratch implementations of:
- LayerNorm -> used by C1 (base), C2, C3, C5
- RMSNorm   -> used by C4

Both are used as Pre-LN: applied to the input BEFORE the sublayer
(attention or FFN), with the residual added around the whole sublayer.
i.e. x = x + Sublayer(Norm(x))   -- not the post-LN variant.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Standard Layer Normalization (Ba et al., 2016).

    y = (x - mean) / sqrt(var + eps) * gamma + beta

    mean/var computed over the last dimension (d_model), per token,
    independent of batch.
    """

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
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    y = x / RMS(x) * gamma,  where RMS(x) = sqrt(mean(x^2) + eps)

    No mean-centering and no learned bias (beta) -- only rescales by the
    root-mean-square, then applies a learned per-dimension scale.
    Cheaper than LayerNorm (no mean subtraction) and is what C4 swaps in.
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


if __name__ == "__main__":
    torch.manual_seed(0)

    d_model = 16
    x = torch.randn(2, 5, d_model) * 3 + 2  # arbitrary mean/scale to make the check meaningful

    # --- LayerNorm checks ---
    ln = LayerNorm(d_model)
    out = ln(x)
    assert out.shape == x.shape

    # with gamma=1, beta=0 (default init), output should have ~0 mean and ~1 std per token
    mean_per_token = out.mean(dim=-1)
    std_per_token = out.std(dim=-1, unbiased=False)
    assert torch.allclose(mean_per_token, torch.zeros_like(mean_per_token), atol=1e-5)
    assert torch.allclose(std_per_token, torch.ones_like(std_per_token), atol=1e-3)
    print("LayerNorm mean~0, std~1 per token check passed")

    # cross-check against torch's own layer_norm as a reference implementation
    ref = torch.nn.functional.layer_norm(x, (d_model,), eps=1e-6)
    assert torch.allclose(out, ref, atol=1e-4)
    print("LayerNorm matches torch.nn.functional.layer_norm reference")

    # --- RMSNorm checks ---
    rms_norm = RMSNorm(d_model)
    out_rms = rms_norm(x)
    assert out_rms.shape == x.shape

    # with gamma=1, output RMS per token should be ~1 (no mean-centering, unlike LayerNorm)
    out_rms_val = torch.sqrt((out_rms ** 2).mean(dim=-1))
    assert torch.allclose(out_rms_val, torch.ones_like(out_rms_val), atol=1e-3)
    print("RMSNorm output RMS~1 per token check passed")

    # RMSNorm should NOT zero-center (unlike LayerNorm) -- confirm output mean is not ~0
    # since input x has a nonzero mean and RMSNorm doesn't subtract it
    out_mean = out_rms.mean(dim=-1)
    assert not torch.allclose(out_mean, torch.zeros_like(out_mean), atol=1e-2)
    print("RMSNorm correctly does NOT zero-center (differs from LayerNorm behavior)")

    print("\nAll checks passed.")
