import torch
import argparse

from modeling.models import Seq2SeqConfig, Seq2Seq
from data.tokenizer import Tokenizer
from data.dataloader import build_seq2seq_dataloader
from training.trainer import Seq2SeqTrainer
import config

def train_seq2seq():
    tokenizer = Tokenizer()
    train_loader = build_seq2seq_dataloader(config.TRAIN_PATH, tokenizer)
    dev_loader = build_seq2seq_dataloader(config.DEV_PATH, tokenizer, shuffle=False)
    model = Seq2Seq(Seq2SeqConfig(
        vocab_size=config.VOCAB_SIZE,
        pad_token_id=tokenizer.pad_id,
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id,
        model_dim=config.MODEL_DIM,
        state_dim=config.STATE_DIM,
        conv_kernel=config.CONV_KERNEL,
        head_dim=config.HEAD_DIM,
        num_groups=config.NUM_GROUPS,
        chunk_size=config.CHUNK_SIZE,
        num_layers=config.NUM_LAYERS,
        device="cuda"
    ))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    trainer = Seq2SeqTrainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        criterion=criterion
    )

    trainer.train()

if __name__ == "__main__":
    if config.TYPE == "seq2seq":
        train_seq2seq()