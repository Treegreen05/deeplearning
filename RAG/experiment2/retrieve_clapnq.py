import argparse
import json

import faiss
from tqdm import tqdm

from shared import Encoder, format_passage, iter_jsonl, read_questions, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-file", required=True)
    parser.add_argument("--index-file", required=True)
    parser.add_argument("--passage-metadata", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-query-length", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    passages = list(iter_jsonl(args.passage_metadata))
    questions = read_questions(args.questions_file, limit=args.max_samples)
    index = faiss.read_index(args.index_file)
    encoder = Encoder(args.model_name, args.adapter_path, args.pooling)

    rows = []
    for start in tqdm(range(0, len(questions), args.batch_size), desc="retrieve"):
        batch = questions[start : start + args.batch_size]
        q_vectors = encoder.encode(
            [row["question"] for row in batch],
            max_length=args.max_query_length,
            batch_size=args.batch_size,
        )
        scores, indices = index.search(q_vectors, args.top_n)
        for row, row_scores, row_indices in zip(batch, scores, indices):
            retrieved = []
            for score, idx in zip(row_scores, row_indices):
                if idx < 0:
                    continue
                passage = passages[int(idx)]
                retrieved.append(
                    {
                        "id": passage["id"],
                        "title": passage.get("title", ""),
                        "text": passage.get("text", ""),
                        "score": float(score),
                    }
                )
            rows.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "gold_doc_ids": row["gold_doc_ids"],
                    "answers": row["answers"],
                    "retrieved": retrieved,
                }
            )

    write_jsonl(args.output, rows)
    print(
        json.dumps(
            {
                "samples": len(rows),
                "top_n": args.top_n,
                "output": args.output,
                "index": args.index_file,
                "model_name": args.model_name,
                "adapter_path": args.adapter_path,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
