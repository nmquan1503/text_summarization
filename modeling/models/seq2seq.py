import torch
import torch.nn as nn
from dataclasses import dataclass

from modeling.modules import Block, RMSNorm

@dataclass
class Seq2SeqConfig:
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


class Seq2Seq(nn.Module):
    def __init__(self, config: Seq2SeqConfig | None = None):
        super().__init__()

        if config is None:
            config = Seq2SeqConfig()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.model_dim)

        self.encoder_layers = nn.ModuleList([
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

        self.decoder_layers = nn.ModuleList([
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
        enc_input_ids: torch.Tensor, 
        enc_input_lengths: torch.Tensor,
        dec_input_ids: torch.Tensor,
        use_cache: bool = False
    ):
        """
        Args: 
            enc_input_ids: (batch_size, enc_seq_len)
            enc_input_lengths: (batch_size,)
            dec_input_ids: (batch_size, dec_seq_len)
        
        Returns:
            logits: (batch_size, dec_seq_len, vocab_size)
        """

        enc_hidden_states = self.embedding(enc_input_ids)
        dec_hidden_states = self.embedding(dec_input_ids)
        
        for enc_layer, dec_layer in zip(self.encoder_layers, self.decoder_layers):
            enc_hidden_states, enc_ssm_hiddens = enc_layer(enc_hidden_states, lengths=enc_input_lengths)
            dec_hidden_states, _ = dec_layer(
                dec_hidden_states,
                ssm_hiddens=enc_ssm_hiddens,
                use_cache=use_cache
            )
        
        hidden_states = self.norm(dec_hidden_states)
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
        for layer in self.decoder_layers:
            hidden_states = layer.step(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    def generate(self, input_ids: torch.Tensor, max_new_tokens=100):
        """
        Args:
            input_ids: (batch_size, seq_len)

        Returns:
            seq_ids: (batch_size, out_seq_len)
        """
        with torch.no_grad():
            batch_size = input_ids.size(0)
            device = input_ids.device
            bos_id = self.config.bos_token_id
            eos_id = self.config.eos_token_id
            pad_id = self.config.pad_token_id

            lengths = (input_ids != pad_id).sum(dim=1)

            seq_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            logits = self.forward(
                enc_input_ids=input_ids, 
                enc_input_lengths=lengths, 
                dec_input_ids=seq_ids, 
                use_cache=True
            )
            logits = logits[:, -1, :]

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