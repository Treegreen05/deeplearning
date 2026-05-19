import argparse
import json
from pathlib import Path

import faiss
from tqdm import tqdm

from shared import Encoder, ensure_parent, format_passage, read_passages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages-file", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--output-index", required=True)
    parser.add_argument("--output-passages", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-passage-length", type=int, default=256)
    parser.add_argument("--max-passages", type=int, default=None)
    args = parser.parse_args()

    passages = read_passages(args.passages_file, limit=args.max_passages)
    if not passages:
        raise ValueError("no passages loaded")

    encoder = Encoder(args.model_name, args.adapter_path, args.pooling)

    index = None
    for start in tqdm(range(0, len(passages), args.batch_size), desc="encode passages"):
        batch = passages[start : start + args.batch_size]
        texts = [format_passage(row) for row in batch]
        vectors = encoder.encode(
            texts,
            max_length=args.max_passage_length,
            batch_size=args.batch_size,
        )
        if index is None:
            index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

    ensure_parent(args.output_index)
    faiss.write_index(index, args.output_index)

    ensure_parent(args.output_passages)
    with open(args.output_passages, "w", encoding="utf-8") as f:
        for row in passages:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "passages": len(passages),
                "index": str(Path(args.output_index)),
                "passage_metadata": str(Path(args.output_passages)),
                "model_name": args.model_name,
                "adapter_path": args.adapter_path,
                "pooling": args.pooling,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
