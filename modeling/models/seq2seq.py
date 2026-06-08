import torch
import torch.nn as nn
from dataclasses import dataclass

from modeling.modules import BiBlock, CrossBlock, RMSNorm

@dataclass
class Seq2SeqConfig:
    vocab_size: int = 32000
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    model_dim: int = 512
    head_dim: int = 8
    expansion_factor: int = 2

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
            BiBlock(
                model_dim=config.model_dim,
                head_dim=config.head_dim,
                expansion_factor=config.expansion_factor,
                dropout_rate=config.dropout_rate,
            )
            for _ in range(config.num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            CrossBlock(
                model_dim=config.model_dim,
                head_dim=config.head_dim,
                expansion_factor=config.expansion_factor,
                dropout_rate=config.dropout_rate,
            )
            for _ in range(config.num_layers)
        ])

        self.norm =  RMSNorm(config.model_dim)

        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, enc_input_ids, enc_attention_mask, dec_input_ids):
        """
        Args: 
            enc_input_ids: (batch_size, enc_seq_len)
            enc_attention_mask: (batch_size, enc_seq_len)
            dec_input_ids: (batch_size, dec_seq_len)
        
        Returns:
            logits: (batch_size, dec_seq_len, vocab_size)
        """

        enc_hidden_states = self.embedding(enc_input_ids)
        dec_hidden_states = self.embedding(dec_input_ids)

        for layer in self.encoder_layers:
            enc_hidden_states = layer(enc_hidden_states)
        
        kv_cache = []

        for layer in self.decoder_layers:
            dec_hidden_states, k, v = layer(dec_hidden_states, enc_hidden_states, enc_attention_mask)
            kv_cache.append({"k": k, "v": v})

        hidden_states = self.norm(dec_hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, enc_hidden_states, kv_cache

    def step(self, input_ids, context, context_attention_mask, kv_cache):
        """
        Args:
            input_ids: (batch_size,)
        
        Returns:
            logits: (batch_size, vocab_size)
        """
        hidden_states = self.embedding(input_ids)
        for layer_idx, layer in enumerate(self.decoder_layers):
            k_cache = kv_cache[layer_idx]["k"]
            v_cache = kv_cache[layer_idx]["v"]
            hidden_states, k_cache, v_cache = layer.step(hidden_states, context, context_attention_mask, k_cache, v_cache)
            kv_cache[layer_idx]["k"] = k_cache
            kv_cache[layer_idx]["v"] = v_cache

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, kv_cache

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

            attention_mask = (input_ids != pad_id)

            seq_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            logits, context, kv_cache = self.forward(input_ids, attention_mask, seq_ids)
            logits = logits[:, -1, :]

            for _ in range(max_new_tokens):
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

                seq_ids = torch.cat([seq_ids, next_token], dim=1)
                finished |= (next_token.squeeze(1) == eos_id)

                if finished.all():
                    break

                logits, kv_cache = self.step(next_token.squeeze(1), context, attention_mask, kv_cache)

            eos_mask = (seq_ids == eos_id)
            first_eos = eos_mask.float().cumsum(dim=1) >= 1
            seq_ids = torch.where(first_eos, eos_id, seq_ids)
            
            return seq_ids