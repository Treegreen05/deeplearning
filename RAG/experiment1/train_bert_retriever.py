import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


def format_passage(ctx):
    title = ctx.get("title") or ""
    text = ctx.get("text") or ""
    return f"{title} {text}".strip()


def choose_ctx(ctxs):
    if not ctxs:
        return None
    return random.choice(ctxs)


class DprTrainDataset(Dataset):
    def __init__(self, path, use_hard_negatives=False):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
        self.use_hard_negatives = use_hard_negatives

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        positive = choose_ctx(row.get("positive_ctxs", []))
        neg_key = "hard_negative_ctxs" if self.use_hard_negatives else "negative_ctxs"
        negative = choose_ctx(row.get(neg_key, [])) or choose_ctx(row.get("negative_ctxs", []))
        if positive is None or negative is None:
            return None
        return {
            "question": row["question"],
            "positive": format_passage(positive),
            "negative": format_passage(negative),
        }


def collate(batch):
    batch = [x for x in batch if x is not None]
    return {
        "questions": [x["question"] for x in batch],
        "positives": [x["positive"] for x in batch],
        "negatives": [x["negative"] for x in batch],
    }


class Retriever(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def encode(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return F.normalize(pooled, p=2, dim=1)


def tokenize(tokenizer, texts, max_length, device):
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in batch.items()}


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="google-bert/bert-base-uncased")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--use-hard-negatives", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-query-length", type=int, default=64)
    parser.add_argument("--max-passage-length", type=int, default=256)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModel.from_pretrained(args.model_name)

    if args.adapter_path:
        base_model = PeftModel.from_pretrained(base_model, args.adapter_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["query", "value"],
        )
        base_model = get_peft_model(base_model, lora_config)

    retriever = Retriever(base_model).to(device)
    retriever.train()
    retriever.model.print_trainable_parameters()

    dataset = DprTrainDataset(args.train_file, use_hard_negatives=args.use_hard_negatives)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(retriever.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * math.ceil(len(dataset) / args.batch_size)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    global_step = 0
    for epoch in range(args.epochs):
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            if not batch["questions"]:
                continue

            q_tok = tokenize(tokenizer, batch["questions"], args.max_query_length, device)
            pos_tok = tokenize(tokenizer, batch["positives"], args.max_passage_length, device)
            neg_tok = tokenize(tokenizer, batch["negatives"], args.max_passage_length, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                q_emb = retriever.encode(**q_tok)
                pos_emb = retriever.encode(**pos_tok)
                neg_emb = retriever.encode(**neg_tok)
                passage_emb = torch.cat([pos_emb, neg_emb], dim=0)
                logits = torch.matmul(q_emb, passage_emb.T) / args.temperature
                labels = torch.arange(q_emb.size(0), device=device)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

    retriever.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_json(output_dir / "training_args.json", vars(args))
    print(f"saved adapter and tokenizer to {output_dir}")


if __name__ == "__main__":
    main()

