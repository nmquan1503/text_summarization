import torch
import torch.nn as nn
from dataclasses import dataclass

from modeling.modules import Block, RMSNorm

@dataclass
class CausalLMConfig:
    vocab_size: int = 32000
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    model_dim: int = 512
    state_dim: int = 16
    conv_kernel: int = 4
    head_dim: int = 8
    num_groups: int = 1
    chunk_size: int = 256

    num_layers: int = 4
    dropout_rate: float = 0.15

    device: str | None = None


class CausalLM(nn.Module):
    def __init__(self, config: CausalLMConfig | None = None):
        super().__init__()

        if config is None:
            config = CausalLMConfig()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.model_dim)

        self.layers = nn.ModuleList([
            Block(
                model_dim=config.model_dim,
                state_dim=config.state_dim,
                conv_kernel=config.conv_kernel,
                head_dim=config.head_dim,
                num_groups=config.num_groups,
                chunk_size=config.chunk_size,
                dropout_rate=config.dropout_rate,
                device=config.device
            )
            for _ in range(config.num_layers)
        ])

        self.norm =  RMSNorm(config.model_dim)

        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(
        self, 
        input_ids: torch.Tensor, 
        lengths: torch.Tensor | None = None,
        use_cache: bool = False
    ):
        """
        Args: 
            input_ids: (batch_size, seq_len)
            lengths: (batch_size,)
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """

        hidden_states = self.embedding(input_ids)
        
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states, lengths=lengths, use_cache=use_cache)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    def step(self, input_ids: torch.Tensor):
        """
        Args:
            input_ids: (batch_size,)
        
        Returns:
            logits: (batch_size, vocab_size)
        """
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer.step(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    def generate(self, input_ids: torch.Tensor, max_new_tokens=100):
        """
        Args:
            input_ids: (batch_size, seq_len)

        Returns:
            seq_ids: (batch_size, seq_len + num_new_token)
        """
        with torch.no_grad():
            batch_size = input_ids.size(0)
            device = input_ids.device
            bos_id = self.config.bos_token_id
            eos_id = self.config.eos_token_id
            pad_id = self.config.pad_token_id

            lengths = (input_ids != pad_id).sum(dim=1)
            last_indices = lengths - 1

            logits = self.forward(input_ids, lengths, use_cache=True)
            logits = logits[:, last_indices]

            seq_ids = input_ids
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_new_tokens):
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

                seq_ids = torch.cat([seq_ids, next_token], dim=1)
                finished |= (next_token.squeeze(1) == eos_id)

                if finished.all():
                    break

                logits = self.step(next_token.squeeze(1))

            eos_mask = (seq_ids == eos_id)
            first_eos = eos_mask.float().cumsum(dim=1) >= 1
            seq_ids = torch.where(first_eos, eos_id, seq_ids)
            
            return seq_ids