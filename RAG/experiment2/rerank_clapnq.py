import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shared import format_passage, iter_jsonl, write_jsonl


class Reranker:
    def __init__(self, model_name, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()

    def score(self, query, passages, batch_size=32, max_length=512):
        scores = []
        with torch.no_grad():
            for start in range(0, len(passages), batch_size):
                batch = passages[start : start + batch_size]
                inputs = self.tokenizer(
                    [query] * len(batch),
                    [format_passage(row) for row in batch],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs).logits
                if logits.ndim == 2 and logits.size(-1) > 1:
                    batch_scores = logits[:, -1]
                else:
                    batch_scores = logits.view(-1)
                scores.extend(batch_scores.detach().cpu().tolist())
        return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-file", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    reranker = Reranker(args.model_name)
    output_rows = []
    for row in tqdm(iter_jsonl(args.retrieval_file, args.max_samples), desc="rerank"):
        candidates = row.get("retrieved", [])[: args.top_n]
        scores = reranker.score(
            row["question"],
            candidates,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        reranked = []
        for passage, score in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True):
            item = dict(passage)
            item["rerank_score"] = float(score)
            reranked.append(item)
        row = dict(row)
        row["retrieved"] = reranked
        output_rows.append(row)

    write_jsonl(args.output, output_rows)
    print(json.dumps({"samples": len(output_rows), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
