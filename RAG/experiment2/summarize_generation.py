import argparse
import json

from shared import iter_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = list(iter_jsonl(args.input))
    result = {
        "input": args.input,
        "samples": len(rows),
        "mode": rows[0].get("mode") if rows else None,
        "k": rows[0].get("k") if rows else None,
        "f1": sum(row.get("f1", 0.0) for row in rows) / len(rows) if rows else 0.0,
        "rouge_l": sum(row.get("rouge_l", 0.0) for row in rows) / len(rows) if rows else 0.0,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
