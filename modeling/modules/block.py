import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

from modeling.modules.rms_norm import RMSNorm
from modeling.modules.ssm import SSM
from modeling.modules.attention import SelectiveMHA
from modeling.modules.feed_forward import SwiGLU

class Block(nn.Module):
    def __init__(
        self,
        model_dim: int = 512,
        state_dim: int = 128,
        conv_kernel: int = 4,
        head_dim: int = 64,
        num_groups: int = 1,
        expansion_factor: int = 2,
        chunk_size: int = 256,
        delta_limit: Tuple[float, float] = (0.0, float("inf")),
        A_init_range: Tuple[int, int] = (1, 16),
        delta_init_limit: Tuple[float, float] = (0.001, 0.1),
        delta_init_floor: float = 1e-4,
        dropout_rate: float = 0.15,
        device="cuda"
    ):
        super().__init__()

        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.ssm = SSM(
            model_dim=model_dim,
            state_dim=state_dim,
            conv_kernel=conv_kernel,
            head_dim=head_dim,
            num_groups=num_groups,
            expansion_factor=expansion_factor,
            chunk_size=chunk_size,
            delta_limit=delta_limit,
            A_init_range=A_init_range,
            delta_init_limit=delta_init_limit,
            delta_init_floor=delta_init_floor,
            dropout_rate=dropout_rate,
            device=device
        )
        self.gate_proj = nn.Linear(model_dim, 1)
        self.mha = SelectiveMHA(model_dim, head_dim)
        self.ffn = SwiGLU(model_dim, model_dim * expansion_factor)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        lengths: torch.Tensor | None = None,
        ssm_hiddens: torch.Tensor | None = None, 
        conv_context: torch.Tensor | None = None,
        use_cache: bool = False
    ):
        """
        Args:
            hidden_states: (batch_size, seq_len, model_dim)
            lengths: (batch_size,)
            ssm_hiddens: (batch_size, inner_dim, state_dim)
            conv_context: (batch_size, hBC_dim, conv_kernel - 1)
        
        Returns: 
            hidden_states: (batch_size, seq_len, model_dim)
            last_ssm_hiddens: (batch_size, inner_dim, state_dim)
        """

        res = hidden_states
        hidden_states = self.norm1(hidden_states)
        ssm_out, last_ssm_hiddens = self.ssm(hidden_states, lengths, ssm_hiddens, conv_context, use_cache)
        gate = torch.sigmoid(self.gate_proj(F.silu(ssm_out)))
        hidden_states = self.mha(hidden_states, lengths, gate, use_cache)
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = res + self.dropout(hidden_states)
        
        return hidden_states, last_ssm_hiddens

    def step(self, hidden_states: torch.Tensor):
        """
        Args: (batch_size, model_dim)
        Returns: (batch_size, model_dim)
        """

        res = hidden_states
        hidden_states = self.norm1(hidden_states)
        ssm_out = self.ssm.step(hidden_states)
        gate = torch.sigmoid(self.gate_proj(F.silu(ssm_out)))
        hidden_states = self.mha.step(hidden_states, gate)
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = res + self.dropout(hidden_states)

        return hidden_states