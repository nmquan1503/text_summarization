import torch
from tqdm import tqdm

from data.tokenizer import Tokenizer
from data.dataloader import build_seq2seq_dataloader
from modeling.models.seq2seq import Seq2SeqConfig, Seq2Seq
import config
from evaluation.metrics import compute_rouge

def _generate_preds_seq2seq():
    tokenizer = Tokenizer()
    test_loader = build_seq2seq_dataloader(config.TEST_PATH, config.TEST_PATH, tokenizer, False)
    device = "cuda"
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
        num_layers=config.NUM_LAYERS
    )).to(device)

    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))

    model.eval()

    all_preds = []
    all_refs = []

    for batch in tqdm(test_loader, desc="Test"):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        seq_ids = model.generate(input_ids, config.MAX_NEW_TOKENS).cpu()
        target_ids = target_ids.cpu()

        for pred, tgt in zip(seq_ids, target_ids):
            pred = pred.tolist()
            tgt = tgt.tolist()

            pred_text = tokenizer.decode(pred)
            tgt_text = tokenizer.decode(tgt)

            all_preds.append(pred_text)
            all_refs.append(tgt_text)
    
    return all_preds, all_refs

def evaluate_seq2seq():
    all_preds, all_refs = _generate_preds_seq2seq()

    results = {}

    results.update(compute_rouge(all_preds, all_refs))

    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
    
if __name__ == "__main__":
    if config.TYPE == "seq2seq":
        evaluate_seq2seq()
