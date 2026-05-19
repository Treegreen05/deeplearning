import argparse
import json

import numpy as np

from shared import iter_jsonl, ndcg_at_k


def labels_for_row(row):
    gold = set(row.get("gold_doc_ids", []))
    return [item.get("id") in gold for item in row.get("retrieved", [])]


def compute_metrics(rows, ks):
    rankings = [labels_for_row(row) for row in rows]
    rankings = [labels for labels in rankings if labels]
    metrics = {"samples": len(rankings)}

    for k in ks:
        metrics[f"recall@{k}"] = float(np.mean([any(labels[:k]) for labels in rankings])) if rankings else 0.0
        metrics[f"ndcg@{k}"] = float(np.mean([ndcg_at_k(labels, k) for labels in rankings])) if rankings else 0.0

    rr = []
    for labels in rankings:
        top = labels[:10]
        hit_positions = np.where(np.array(top, dtype=bool))[0]
        rr.append(0.0 if len(hit_positions) == 0 else 1.0 / float(hit_positions[0] + 1))
    metrics["mrr@10"] = float(np.mean(rr)) if rr else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ks", default="1,3,5,10")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    rows = list(iter_jsonl(args.input, args.max_samples))
    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    result = {
        "input": args.input,
        "metrics": compute_metrics(rows, ks),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
