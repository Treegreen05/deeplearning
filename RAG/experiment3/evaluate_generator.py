import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiment2"))
from generate_answers import build_prompt
from shared import best_metric_against_answers, iter_jsonl, read_passages, read_questions, rouge_l, token_f1, write_jsonl


class Generator:
    def __init__(self, model_name, adapter_path=None, device=None, fp16=True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = torch.float16 if fp16 and self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.to(self.device).eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt, max_input_length=1536, max_new_tokens=128):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def load_retrieval_map(path):
    if not path:
        return {}
    return {row["id"]: row for row in iter_jsonl(path)}


def load_passage_map(path):
    return {row["id"]: row for row in read_passages(path)}


def select_passages(mode, row, passage_map, retrieval_map, k):
    if mode == "gold":
        passages = []
        for pid in row.get("gold_doc_ids", []):
            if pid in passage_map:
                passages.append(passage_map[pid])
        return passages
    if mode == "reranked":
        retrieved = retrieval_map.get(row["id"])
        if retrieved is None:
            raise KeyError(f"missing retrieval result for question id {row['id']}")
        return retrieved.get("retrieved", [])[:k]
    raise ValueError(f"unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-file", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["gold", "reranked"], required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--passages-file", default=None)
    parser.add_argument("--retrieval-file", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-input-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    if args.mode == "gold" and not args.passages_file:
        raise ValueError("--passages-file is required for gold mode")
    if args.mode == "reranked" and not args.retrieval_file:
        raise ValueError("--retrieval-file is required for reranked mode")

    questions = read_questions(args.questions_file, limit=args.max_samples)
    passage_map = load_passage_map(args.passages_file) if args.passages_file else {}
    retrieval_map = load_retrieval_map(args.retrieval_file)
    generator = Generator(args.model_name, args.adapter_path, fp16=not args.no_fp16)

    output_rows = []
    skipped = 0
    for row in tqdm(questions, desc=f"evaluate {args.mode}"):
        passages = select_passages(args.mode, row, passage_map, retrieval_map, args.k)
        if args.mode == "gold" and not passages:
            skipped += 1
            continue
        prompt = build_prompt(row["question"], passages)
        prediction = generator.generate(
            prompt,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
        )
        output_rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answers": row["answers"],
                "mode": args.mode,
                "k": args.k if args.mode == "reranked" else None,
                "adapter_path": args.adapter_path,
                "prediction": prediction,
                "f1": best_metric_against_answers(prediction, row["answers"], token_f1),
                "rouge_l": best_metric_against_answers(prediction, row["answers"], rouge_l),
                "passages": [
                    {
                        "id": p.get("id"),
                        "title": p.get("title", ""),
                        "text": p.get("text", ""),
                    }
                    for p in passages
                ],
            }
        )

    write_jsonl(args.output, output_rows)
    f1 = sum(row["f1"] for row in output_rows) / len(output_rows) if output_rows else 0.0
    rouge = sum(row["rouge_l"] for row in output_rows) / len(output_rows) if output_rows else 0.0
    result = {
        "samples": len(output_rows),
        "skipped": skipped,
        "mode": args.mode,
        "k": args.k if args.mode == "reranked" else None,
        "model_name": args.model_name,
        "adapter_path": args.adapter_path,
        "f1": f1,
        "rouge_l": rouge,
        "output": args.output,
    }
    with open(str(Path(args.output).with_suffix(".metrics.json")), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
