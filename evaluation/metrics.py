from rouge_score import rouge_scorer

def compute_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(preds, refs):
        s = scorer.score(ref, pred)
        for k in scores:
            scores[k].append(s[k].fmeasure)
    
    return {
        k: sum(v) / len(v)
        for k, v in scores.items()
    }