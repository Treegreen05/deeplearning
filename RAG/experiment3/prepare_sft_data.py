import argparse
import json
from pathlib import Path

from tqdm import tqdm

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiment2"))
from generate_answers import build_prompt
from shared import iter_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    rows = []
    skipped = 0
    for row in tqdm(iter_jsonl(args.retrieval_file, args.max_samples), desc="prepare sft"):
        answers = row.get("answers", [])
        if not answers:
            skipped += 1
            continue
        passages = row.get("retrieved", [])[: args.k]
        if not passages:
            skipped += 1
            continue
        rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "prompt": build_prompt(row["question"], passages),
                "answer": answers[0],
                "answers": answers,
                "passages": passages,
            }
        )

    write_jsonl(args.output, rows)
    print(json.dumps({"samples": len(rows), "skipped": skipped, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
