import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial

from data.tokenizer import Tokenizer
from data.dataset import Seq2SeqDataset
import config

def seq2seq_collate_fn(batch, pad_id):
    input_ids = [item["input_ids"] for item in batch]
    target_ids = [item["target_ids"] for item in batch]

    input_lengths = torch.tensor([ip.size(0) for ip in input_ids], dtype=torch.long)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    target_ids = pad_sequence(target_ids, batch_first=True, padding_value=pad_id)

    return {
        "input_ids": input_ids,
        "input_lengths": input_lengths,
        "target_ids": target_ids
    }

def build_seq2seq_dataloader(dataset_path: str, tokenizer: Tokenizer, shuffle=True):
    dataset = Seq2SeqDataset(dataset_path, tokenizer)
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=seq2seq_collate_fn
    )