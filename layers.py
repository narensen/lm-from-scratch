import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO 1:
        1. Upcast x to float32 for numerical stability under bf16/fp16.
        2. Compute rms = sqrt(mean(x^2, dim=-1, keepdim=True) + eps).
        3. Normalize x / rms, cast back to original dtype, and multiply by self.weight.
        """
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
        raise NotImplementedError


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        # TODO 3a: Precompute inv_freq = 1.0 / (theta ** (arange(0, head_dim, 2) / head_dim))
        # Build cos/sin tables of shape (max_seq_len, head_dim // 2) and register_buffer(..., persistent=False)
        raise NotImplementedError

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., seq_len, head_dim)
        positions: (seq_len,) or (batch, seq_len) integer token indices
        TODO 3b:
        Split x into pairs (x_even, x_odd) — or first/second half — rotate by cos/sin
        at `positions` in float32, and return tensor with identical shape & dtype as x.
        """
        raise NotImplementedError


if __name__ == "__main__":
    torch.manual_seed(0)

    d_model, eps = 64, 1e-6

    mine = RMSNorm(d_model, eps)
    ref = nn.RMSNorm(d_model, eps)

    x1 = torch.randn(2, 16, d_model, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1, y2 = mine(x1), ref(x2)
    assert torch.allclose(y1, y2, atol=1e-6), f"Forward diff: {(y1 - y2).abs().max()}"

    y1.sum().backward()
    y2.sum().backward()
    assert torch.allclose(x1.grad, x2.grad , atol=1e-6)
    assert torch.allclose(mine.weight.grad, ref.weight.grad, atol=1e-6)

    x_bf16 = torch.randn(2, 16, d_model, dtype=torch.bfloat16) * 50.0
    y_bf16 = mine(x_bf16)
    assert y_bf16.dtype == torch.bfloat16
    rms_out = y_bf16.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms_out, torch.ones_like(rms_out), atol=5e-2), f"Bad RMS: {rms_out.mean()}"

    print("RMSNorm passed forward, backward, and bf16 checks!")