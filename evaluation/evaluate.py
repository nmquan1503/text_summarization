import torch
from tqdm import tqdm

from data.tokenizer import Tokenizer
from data.dataloader import auto_dataloader
from modeling.models import auto_model
import config
from evaluation.metrics import compute_rouge

def _generate_preds_seq2seq(model, tokenizer, data_loader):
    all_preds = []
    all_refs = []

    for batch in tqdm(data_loader, desc="Test"):
        input_ids = batch["input_ids"].to("cuda")
        target_ids = batch["target_ids"].to("cuda")

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

def _generate_preds_causal_lm(model, tokenizer, data_loader):
    all_preds = []
    all_refs = []

    for batch in tqdm(data_loader, desc="Test"):
        gen_input_ids = batch["gen_input_ids"].to("cuda")
        target_ids = batch["target_ids"]

        seq_ids = model.generate(gen_input_ids, config.MAX_NEW_TOKENS).cpu()

        for pred, tgt in zip(seq_ids, target_ids):
            pred = pred.tolist()

            pred_text = tokenizer.decode(pred)
            tgt_text = tokenizer.decode(tgt)

            all_preds.append(pred_text)
            all_refs.append(tgt_text)
    
    return all_preds, all_refs

def _generate_preds():
    tokenizer = Tokenizer()
    test_loader = auto_dataloader(tokenizer, "test")
    device = "cuda"
    model = auto_model()
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    if config.TYPE == "seq2seq":
        return _generate_preds_seq2seq(model, tokenizer, test_loader)
    elif config.TYPE == "causal_lm":
        return _generate_preds_causal_lm(model, tokenizer, test_loader)
    else:
        raise ValueError(f"Don't support {config.TYPE} generation.")

def evaluate():
    all_preds, all_refs = _generate_preds()

    results = {}

    results.update(compute_rouge(all_preds, all_refs))

    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
    
if __name__ == "__main__":
    evaluate()
