import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def format_passage(ctx):
    title = ctx.get("title") or ""
    text = ctx.get("text") or ""
    return f"{title} {text}".strip()


def iter_jsonl(path, limit=None):
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
                count += 1
                if limit is not None and count >= limit:
                    break


def unique_candidates(row):
    seen = set()
    candidates = []
    for is_positive, key in [
        (True, "positive_ctxs"),
        (False, "negative_ctxs"),
        (False, "hard_negative_ctxs"),
    ]:
        for ctx in row.get(key, []):
            text = format_passage(ctx)
            if not text:
                continue
            pid = ctx.get("passage_id")
            marker = pid if pid is not None else text
            if marker in seen:
                if is_positive:
                    for item in candidates:
                        if item["marker"] == marker:
                            item["is_positive"] = True
                continue
            seen.add(marker)
            candidates.append({"text": text, "is_positive": is_positive, "marker": marker})
    return candidates


class Encoder:
    def __init__(self, model_name, adapter_path=None, pooling="mean", device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_name)
        model = AutoModel.from_pretrained(model_name)
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.to(self.device).eval()
        self.pooling = pooling

    def encode(self, texts, max_length=256, batch_size=64):
        vectors = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                inputs = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                if self.pooling == "cls":
                    pooled = outputs.last_hidden_state[:, 0]
                else:
                    token_embeddings = outputs.last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
                    pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                pooled = F.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.cpu().numpy())
        return np.concatenate(vectors, axis=0)


def rank_sample(encoder, row, batch_size, max_query_length, max_passage_length):
    candidates = unique_candidates(row)
    if not candidates or not any(x["is_positive"] for x in candidates):
        return None

    q_emb = encoder.encode([row["question"]], max_length=max_query_length, batch_size=1)[0]
    p_emb = encoder.encode(
        [x["text"] for x in candidates],
        max_length=max_passage_length,
        batch_size=batch_size,
    )
    scores = p_emb @ q_emb
    order = np.argsort(-scores)
    labels = np.array([candidates[i]["is_positive"] for i in order], dtype=bool)
    return labels


def compute_metrics(rankings):
    n = len(rankings)
    if n == 0:
        return {"samples": 0}

    metrics = {"samples": n}
    for k in [10, 20, 30]:
        metrics[f"recall@{k}"] = float(np.mean([labels[:k].any() for labels in rankings]))

    rr = []
    for labels in rankings:
        top = labels[:10]
        hit_positions = np.where(top)[0]
        rr.append(0.0 if len(hit_positions) == 0 else 1.0 / float(hit_positions[0] + 1))
    metrics["mrr@10"] = float(np.mean(rr))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-file", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-query-length", type=int, default=64)
    parser.add_argument("--max-passage-length", type=int, default=256)
    args = parser.parse_args()

    encoder = Encoder(args.model_name, args.adapter_path, args.pooling)
    rankings = []
    for row in tqdm(iter_jsonl(args.dev_file, args.max_samples), desc="evaluate"):
        labels = rank_sample(
            encoder,
            row,
            batch_size=args.batch_size,
            max_query_length=args.max_query_length,
            max_passage_length=args.max_passage_length,
        )
        if labels is not None:
            rankings.append(labels)

    metrics = compute_metrics(rankings)
    result = {
        "model_name": args.model_name,
        "adapter_path": args.adapter_path,
        "pooling": args.pooling,
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

