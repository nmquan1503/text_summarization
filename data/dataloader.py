import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from data.tokenizer import Tokenizer
from data.dataset import auto_dataset
import config

def seq2seq_collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    target_ids = [item["target_ids"] for item in batch]

    input_lengths = torch.tensor([ip.size(0) for ip in input_ids], dtype=torch.long)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=config.PAD_ID)
    target_ids = pad_sequence(target_ids, batch_first=True, padding_value=config.PAD_ID)

    return {
        "input_ids": input_ids,
        "input_lengths": input_lengths,
        "target_ids": target_ids
    }

def causal_lm_collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    gen_input_ids = [item["gen_input_ids"] for item in batch]
    tgt_ids = [item["tgt_ids"] for item in batch]

    return {
        "lengths": torch.tensor([ip.size(0) for ip in gen_input_ids], dtype=torch.long),
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=config.PAD_ID),
        "labels": pad_sequence(labels, batch_first=True, padding_value=config.PAD_ID),
        "tgt_ids": tgt_ids
    }

def auto_dataloader(tokenizer: Tokenizer, mode="train"):
    dataset = auto_dataset(tokenizer, mode)
    if config.TYPE == "seq2seq":
        collate_fn = seq2seq_collate_fn
    elif config.TYPE == "causal_lm":
        collate_fn = causal_lm_collate_fn
    else:
        raise ValueError(f"Don't support {config.TYPE} dataloader.")
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(mode == "train"),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=collate_fn
    )