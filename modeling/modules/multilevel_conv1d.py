import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLevelConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, radius):
        super().__init__()
        
        self.radius = radius

        self.causal_conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=radius + 1,
            padding=radius
        )
        self.right_linears = nn.ModuleList([
            nn.Linear(in_channels, out_channels)
            for _ in range(radius)
        ])

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, in_channels)
        
        Returns:
            out: (batch_size, radius + 1, seq_len, out_channels)    # full -> causal
        """
        seq_len = x.shape[1]
        base = self.causal_conv1d(x.transpose(1, 2)).transpose(1, 2)
        base = base[:, :seq_len]

        outs = []
        outs.append(base)
        cur = base
        for i in range(self.radius):
            shifted = x[:, i+1:]
            right = self.right_linears[i](shifted)
            right = F.pad(right, (0, 0, 0, i+1))
            cur = cur + right
            outs.append(cur)
        
        return torch.stack(outs[::-1], dim=1)