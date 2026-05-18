import argparse
import json
from pathlib import Path

from tqdm import tqdm


def parse_limit(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"all", "full", "none", "0", "-1"}:
        return None
    limit = int(text)
    if limit < 0:
        return None
    return limit


def maybe_slice(items, limit):
    if limit is None:
        return items
    return items[:limit]


def iter_json_array(path, chunk_size=1024 * 1024):
    """Stream objects from a top-level JSON array without loading the file."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer += chunk

            while True:
                buffer = buffer.lstrip()
                if not started:
                    if not buffer:
                        break
                    if buffer[0] != "[":
                        raise ValueError(f"{path} is not a JSON array")
                    buffer = buffer[1:]
                    started = True
                    continue

                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue
                if buffer[0] == "]":
                    return

                try:
                    obj, idx = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                yield obj
                buffer = buffer[idx:]


def compact_train_row(row, max_positives, max_negatives, max_hard_negatives):
    return {
        "dataset": row.get("dataset"),
        "question": row.get("question"),
        "answers": row.get("answers", []),
        "positive_ctxs": maybe_slice(row.get("positive_ctxs", []), max_positives),
        "negative_ctxs": maybe_slice(row.get("negative_ctxs", []), max_negatives),
        "hard_negative_ctxs": maybe_slice(row.get("hard_negative_ctxs", []), max_hard_negatives),
    }


def write_jsonl(src, dst, limit=None, transform=None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(dst, "w", encoding="utf-8") as out:
        for obj in tqdm(iter_json_array(src), desc=f"write {dst.name}"):
            if transform is not None:
                obj = transform(obj)
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--dev-file", required=True)
    parser.add_argument("--out-dir", default="experiment1/data")
    parser.add_argument("--max-train", type=parse_limit, default=None)
    parser.add_argument("--max-dev", type=parse_limit, default=None)
    parser.add_argument("--max-train-positives", type=parse_limit, default=None)
    parser.add_argument("--max-train-negatives", type=parse_limit, default=None)
    parser.add_argument("--max-train-hard-negatives", type=parse_limit, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    train_count = write_jsonl(
        Path(args.train_file),
        out_dir / "train.jsonl",
        args.max_train,
        transform=lambda row: compact_train_row(
            row,
            args.max_train_positives,
            args.max_train_negatives,
            args.max_train_hard_negatives,
        ),
    )
    dev_count = write_jsonl(Path(args.dev_file), out_dir / "dev.jsonl", args.max_dev)

    print(json.dumps({"train": train_count, "dev": dev_count, "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
