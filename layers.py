import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        inv_rms = torch.rsqrt(torch.mean((x.to((torch.float32))**2), dim=-1, keepdim=True) + self.eps)
        x = (self.weight * x * inv_rms).to(x.dtype)
        return x


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        if d_ff is None:
            # Standard ~8/3 * d_model rounded up to a multiple of 64 for tensor-core alignment
            d_ff = int(8 / 3 * d_model)
            d_ff = ((d_ff + 63) // 64) * 64
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # up
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO 2: Compute w2(SiLU(w1(x)) * w3(x)).
        """

        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return x


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        # TODO 3a: Precompute inv_freq = 1.0 / (theta ** (arange(0, head_dim, 2) / head_dim))
        # Build cos/sin tables of shape (max_seq_len, head_dim // 2) and register_buffer(..., persistent=False)

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2) / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., seq_len, head_dim)
        positions: (seq_len,) or (batch, seq_len) integer token indices
        TODO 3b:
        Split x into pairs (x_even, x_odd) — or first/second half — rotate by cos/sin
        at `positions` in float32, and return tensor with identical shape & dtype as x.
        """
        in_dtype = x.dtype
        x = x.float()
        cos = self.cos[positions]
        sin = self.sin[positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos

        out = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
        return out.to(in_dtype)

        raise NotImplementedError


if __name__ == "__main__":
    rope = RoPE(head_dim=64, max_seq_len=128)
    q = torch.randn(1, 1, 64)
    k = torch.randn(1, 1, 64)
    dot_0_5 = (rope(q, torch.tensor([0])) * rope(k, torch.tensor([5]))).sum()
    dot_17_22 = (rope(q, torch.tensor([17])) * rope(k, torch.tensor([22]))).sum()
    assert torch.allclose(dot_0_5, dot_17_22, atol=1e-5), f"RoPE broke shift invariance: {dot_0_5} vs {dot_17_22}"

    print("All Day 1 primitive checks passed!")