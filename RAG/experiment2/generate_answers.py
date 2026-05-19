import argparse
import json
import random

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared import (
    best_metric_against_answers,
    choose_random_passages,
    format_passage,
    iter_jsonl,
    read_passages,
    read_questions,
    rouge_l,
    token_f1,
    write_jsonl,
)


RAG_TEMPLATE = """You are given a question and several passages.
Answer the question using only the information in the passages.
If the passages do not contain enough information, answer "I don't know".

Question:
{question}

Passages:
{passages}

Answer:"""


NO_RAG_TEMPLATE = """Answer the following question.
If you do not know the answer, answer "I don't know".

Question:
{question}

Answer:"""


def build_passage_block(passages):
    lines = []
    for idx, row in enumerate(passages, start=1):
        title = row.get("title", "")
        text = row.get("text", "")
        prefix = f"[{idx}]"
        if title:
            lines.append(f"{prefix} {title}: {text}")
        else:
            lines.append(f"{prefix} {text}")
    return "\n".join(lines)


def build_prompt(question, passages):
    if not passages:
        return NO_RAG_TEMPLATE.format(question=question)
    return RAG_TEMPLATE.format(question=question, passages=build_passage_block(passages))


class Generator:
    def __init__(self, model_name, device=None, fp16=True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = torch.float16 if fp16 and self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device).eval()
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


def load_retrieval_map(path, limit=None):
    if not path:
        return {}
    return {row["id"]: row for row in iter_jsonl(path, limit)}


def select_passages(mode, question_row, retrieval_map, all_passages, k, rng):
    if mode == "no_rag":
        return []
    if mode == "random":
        return choose_random_passages(all_passages, k, rng)
    if mode in {"rag", "reranked"}:
        row = retrieval_map.get(question_row["id"])
        if row is None:
            raise KeyError(f"missing retrieval result for question id {question_row['id']}")
        return row.get("retrieved", [])[:k]
    raise ValueError(f"unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-file", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--mode", choices=["no_rag", "random", "rag", "reranked"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--retrieval-file", default=None)
    parser.add_argument("--passages-file", default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-input-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    if args.mode in {"rag", "reranked"} and not args.retrieval_file:
        raise ValueError("--retrieval-file is required for rag/reranked modes")
    if args.mode == "random" and not args.passages_file:
        raise ValueError("--passages-file is required for random mode")

    questions = read_questions(args.questions_file, limit=args.max_samples)
    retrieval_map = load_retrieval_map(args.retrieval_file)
    all_passages = read_passages(args.passages_file) if args.passages_file else []
    rng = random.Random(args.seed)
    generator = Generator(args.model_name, fp16=not args.no_fp16)

    output_rows = []
    for row in tqdm(questions, desc=f"generate {args.mode}"):
        passages = select_passages(args.mode, row, retrieval_map, all_passages, args.k, rng)
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
                "k": args.k if passages else None,
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
        "mode": args.mode,
        "k": args.k,
        "f1": f1,
        "rouge_l": rouge,
        "output": args.output,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
