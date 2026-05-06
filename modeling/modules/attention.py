import torch
import torch.nn as nn
import torch.nn.functional as F

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(seq_len_or_positions, dim, device, mode="seq"):
    """
    if mode == "seq":
        Returns: 
            cos, sin: (seq_len, dim)

    if mode == "pos":
        Returns:
            cos, sin: (batch_size, dim)
    """
    assert dim % 2 == 0

    half_dim = dim // 2
    freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))

    if mode == 'seq':
        seq_len = seq_len_or_positions
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        angles = torch.einsum("i,j->ij", pos, freq)
    elif mode == 'pos':
        positions = seq_len_or_positions.float()
        angles = positions.unsqueeze(-1) * freq.unsqueeze(0) 
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    return cos, sin


def apply_rotary(q, k, cos, sin, mode="seq"):
    if mode == "seq":
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    elif mode == "pos":
        cos = cos[:, None, None, :]
        sin = sin[:, None, None, :]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin

    return q_rot, k_rot

def build_attn_matrix(attn_matrix, gate):
    """
    attn_matrix: (batch_size, num_heads, seq_len, seq_len)
    gate: (batch_size, k + 1, seq_len)
    """
    batch_size, num_heads, seq_len, _ = attn_matrix.shape
    K = gate.shape[1] - 1
    device = attn_matrix.device

    idx = torch.arange(seq_len, device=device)
    rel = idx[:, None] - idx[None, :]
    future_mask = rel < 0
    dist = rel.clamp(min=0, max=K)
    level = K - dist
    gate = gate.unsqueeze(1)
    gate = gate.clamp(min=1e-12)
    level_idx = level.unsqueeze(0).unsqueeze(0)
    level_idx = level_idx.expand(batch_size, num_heads, seq_len, seq_len)
    gate = torch.gather(
        gate.expand(batch_size, num_heads, K + 1, seq_len),
        dim=2,
        index=level_idx
    )
    attn_matrix = attn_matrix + torch.log(gate)
    attn_matrix = attn_matrix.masked_fill(future_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    return attn_matrix


class SelectiveMHA(nn.Module):
    def __init__(self, dim, head_dim):
        super().__init__()
        assert dim % head_dim == 0

        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim

        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)

        self.alpha = nn.Parameter(torch.tensor(1.0))

        self._k_rot = None
        self._v = None 
        self._attn_bias = None
        self._valid_mask = None

    def forward(self, hidden_states, lengths, gate, use_cache=False, gate_threshold=0.0):
        """
        Args:
            hidden_states: (batch_size, seq_len, dim)
            lengths: (batch_size)
            gate: (batch_size, mlconv_radius + 1, seq_len)
        
        Returns:
            hidden_states: (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = build_rope_cache(seq_len, self.head_dim, hidden_states.device)
        q, k = apply_rotary(q, k, cos, sin)

        scale = self.head_dim ** 0.5
        attn = (q @ k.transpose(-2, -1)) / scale

        attn = build_attn_matrix(attn, gate)

        attn = F.softmax(attn, dim=-1)

        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        # if use_cache:
        #     self._k_rot = k
        #     self._v = v
        #     self._attn_bias = attn_bias
        #     idx = torch.arange(seq_len, device=hidden_states.device)[None, :]
        #     self._valid_mask = (idx < lengths[:, None]).float()

        return self.out_proj(out)

    def step(self, hidden_states, gate, gate_threshold=0.0):
        """
        Args:
            hidden_states: (batch_size, model_dim)
            gate: (batch_size,)
        
        Returns:
            hidden_states: (batch_size, model_dim)
        """

        batch_size, _ = hidden_states.shape
        device = hidden_states.device

        gate[gate < gate_threshold] = 1e-12

        current_lengths = self._valid_mask.sum(dim=1)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = build_rope_cache(current_lengths, self.head_dim, device, mode='pos')
        q, k = apply_rotary(q, k, cos, sin, mode='pos')

        self._k_rot = torch.cat([self._k_rot, k], dim=2)
        self._v = torch.cat([self._v, v], dim=2)

        new_valid = torch.ones(batch_size, 1, device=device)
        self._valid_mask = torch.cat([self._valid_mask, new_valid], dim=1)

        new_bias = self.alpha * torch.log(gate.clamp(min=1e-12)).unsqueeze(1)
        self._attn_bias = torch.cat([self._attn_bias, new_bias], dim=1)
        
        scale = self.head_dim ** 0.5
        attn = (q @ self._k_rot.transpose(-2, -1)) / scale

        attn = attn + self._attn_bias[:, None, None, :]

        attn = attn.masked_fill(
            self._valid_mask[:, None, None, :] == 0,
            float("-inf")
        )

        attn = F.softmax(attn, dim=-1)
        out = attn @ self._v
        out = out.transpose(1, 2).contiguous().view(batch_size, self.dim)

        return self.out_proj(out)
