#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --time=04:00:00

set -euo pipefail

echo "===== Experiment 1 job started ====="
date
hostname
pwd
nvidia-smi || true

# Change these two lines to match your account and environment.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rag-exp

python --version

python experiment1/make_dpr_subset.py \
  --train-file DPR-NQ/biencoder-nq-train.json \
  --dev-file DPR-NQ/biencoder-nq-dev.json \
  --out-dir experiment1/data \
  --max-train 20000 \
  --max-dev 1000 \
  --max-train-negatives 8 \
  --max-train-hard-negatives 8

python experiment1/evaluate_dpr_candidates.py \
  --dev-file experiment1/data/dev.jsonl \
  --model-name google-bert/bert-base-uncased \
  --pooling mean \
  --output experiment1/results/raw_bert_dev.json

python experiment1/train_bert_retriever.py \
  --train-file experiment1/data/train.jsonl \
  --output-dir experiment1/checkpoints/bert_lora \
  --model-name google-bert/bert-base-uncased \
  --epochs 1 \
  --batch-size 16 \
  --lr 2e-5 \
  --fp16

python experiment1/evaluate_dpr_candidates.py \
  --dev-file experiment1/data/dev.jsonl \
  --model-name google-bert/bert-base-uncased \
  --adapter-path experiment1/checkpoints/bert_lora \
  --pooling mean \
  --output experiment1/results/bert_lora_dev.json

python experiment1/train_bert_retriever.py \
  --train-file experiment1/data/train.jsonl \
  --output-dir experiment1/checkpoints/bert_lora_hn \
  --model-name google-bert/bert-base-uncased \
  --adapter-path experiment1/checkpoints/bert_lora \
  --use-hard-negatives \
  --epochs 1 \
  --batch-size 16 \
  --lr 2e-5 \
  --fp16

python experiment1/evaluate_dpr_candidates.py \
  --dev-file experiment1/data/dev.jsonl \
  --model-name google-bert/bert-base-uncased \
  --adapter-path experiment1/checkpoints/bert_lora_hn \
  --pooling mean \
  --output experiment1/results/bert_lora_hn_dev.json

python experiment1/evaluate_dpr_candidates.py \
  --dev-file experiment1/data/dev.jsonl \
  --model-name BAAI/bge-base-en-v1.5 \
  --pooling cls \
  --output experiment1/results/bge_base_en_v15_dev.json

echo "===== Experiment 1 job finished ====="
date
