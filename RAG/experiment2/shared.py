import csv
import json
import math
import random
import re
import string
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def iter_jsonl(path, limit=None):
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
                count += 1
                if limit is not None and count >= limit:
                    break


def write_jsonl(path, rows):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_passages(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "id": row["id"],
                    "title": row.get("title", ""),
                    "text": row.get("text", ""),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def parse_gold_doc_ids(value):
    value = (value or "").strip()
    if not value:
        return []
    return [x for x in re.split(r"[,\s;]+|::", value) if x]


def parse_answers(value):
    value = (value or "").strip()
    if not value:
        return []
    return [x.strip() for x in value.split("::") if x.strip()]


def read_questions(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "gold_doc_ids": parse_gold_doc_ids(row.get("doc-id-list", "")),
                    "answers": parse_answers(row.get("answers", "")),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def format_passage(row):
    title = row.get("title") or ""
    text = row.get("text") or ""
    return f"{title} {text}".strip()


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
                vectors.append(pooled.cpu().numpy().astype("float32"))
        return np.concatenate(vectors, axis=0)


def normalize_answer(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for tok in pred_tokens:
        common[tok] = common.get(tok, 0) + 1
    overlap = 0
    for tok in gold_tokens:
        if common.get(tok, 0) > 0:
            overlap += 1
            common[tok] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_length(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0]
        for j, y in enumerate(b, start=1):
            curr.append(prev[j - 1] + 1 if x == y else max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    beta = precision / (recall + 1e-12)
    return ((1 + beta * beta) * precision * recall) / (recall + beta * beta * precision + 1e-12)


def best_metric_against_answers(prediction, answers, metric_fn):
    if not answers:
        return 0.0
    return max(metric_fn(prediction, answer) for answer in answers)


def ndcg_at_k(labels, k):
    labels = labels[:k]
    dcg = sum((1.0 if hit else 0.0) / math.log2(rank + 2) for rank, hit in enumerate(labels))
    ideal_hits = min(sum(labels), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return 0.0 if idcg == 0 else dcg / idcg


def choose_random_passages(passages, k, rng):
    if k <= 0:
        return []
    if k >= len(passages):
        return list(passages)
    return rng.sample(passages, k)
