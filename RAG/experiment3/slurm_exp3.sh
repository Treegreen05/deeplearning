#!/usr/bin/env bash
#SBATCH --job-name=rag-exp3
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --time=24:00:00

set -euo pipefail

echo "===== Experiment 3 generator SFT job started ====="
date
hostname

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
echo "project root: $(pwd)"
nvidia-smi || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rag-exp

export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

PASSAGES_FILE="CLAPNQ/passages/passages.tsv"
TRAIN_FILE="CLAPNQ/train/question_train_answerable.tsv"
DEV_FILE="CLAPNQ/dev/question_dev_answerable.tsv"

BERT_MODEL="${BERT_MODEL:-models/bert-base-uncased}"
RETRIEVER_ADAPTER="${RETRIEVER_ADAPTER:-experiment1/checkpoints/bert_lora_hn_full}"
RERANKER_MODEL="${RERANKER_MODEL:-models/bge-reranker-base}"
GENERATOR_MODEL="${GENERATOR_MODEL:-models/Qwen2.5-1.5B-Instruct}"
GENERATOR_ADAPTER="${GENERATOR_ADAPTER:-experiment3/checkpoints/qwen_lora}"

TOP_N="${TOP_N:-30}"
BEST_K="${BEST_K:-5}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-5}"
MAX_SAMPLES_ARG=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  MAX_SAMPLES_ARG=(--max-samples "${MAX_SAMPLES}")
fi

mkdir -p experiment2/index experiment2/results experiment3/data experiment3/checkpoints experiment3/results

if [[ ! -f experiment2/index/clapnq.faiss || ! -f experiment2/index/passages.jsonl ]]; then
  python experiment2/build_faiss_index.py \
    --passages-file "${PASSAGES_FILE}" \
    --model-name "${BERT_MODEL}" \
    --adapter-path "${RETRIEVER_ADAPTER}" \
    --pooling mean \
    --output-index experiment2/index/clapnq.faiss \
    --output-passages experiment2/index/passages.jsonl \
    --batch-size 128
fi

if [[ ! -f experiment3/data/train_retrieved_top${TOP_N}.jsonl ]]; then
  python experiment2/retrieve_clapnq.py \
    --questions-file "${TRAIN_FILE}" \
    --index-file experiment2/index/clapnq.faiss \
    --passage-metadata experiment2/index/passages.jsonl \
    --model-name "${BERT_MODEL}" \
    --adapter-path "${RETRIEVER_ADAPTER}" \
    --pooling mean \
    --top-n "${TOP_N}" \
    --output "experiment3/data/train_retrieved_top${TOP_N}.jsonl" \
    "${MAX_SAMPLES_ARG[@]}"
fi

if [[ ! -f experiment3/data/train_reranked_top${TOP_N}.jsonl ]]; then
  python experiment2/rerank_clapnq.py \
    --retrieval-file "experiment3/data/train_retrieved_top${TOP_N}.jsonl" \
    --model-name "${RERANKER_MODEL}" \
    --top-n "${TOP_N}" \
    --output "experiment3/data/train_reranked_top${TOP_N}.jsonl" \
    "${MAX_SAMPLES_ARG[@]}"
fi

python experiment3/prepare_sft_data.py \
  --retrieval-file "experiment3/data/train_reranked_top${TOP_N}.jsonl" \
  --k "${BEST_K}" \
  --output "experiment3/data/qwen_sft_train_k${BEST_K}.jsonl" \
  "${MAX_SAMPLES_ARG[@]}"

python experiment3/train_qwen_lora.py \
  --train-file "experiment3/data/qwen_sft_train_k${BEST_K}.jsonl" \
  --model-name "${GENERATOR_MODEL}" \
  --output-dir "${GENERATOR_ADAPTER}" \
  --epochs "${EPOCHS}" \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --lr "${LR}" \
  --max-input-length 1536 \
  --max-output-length 256 \
  --fp16 \
  --gradient-checkpointing

if [[ ! -f experiment2/results/reranked_top${TOP_N}.jsonl ]]; then
  python experiment2/retrieve_clapnq.py \
    --questions-file "${DEV_FILE}" \
    --index-file experiment2/index/clapnq.faiss \
    --passage-metadata experiment2/index/passages.jsonl \
    --model-name "${BERT_MODEL}" \
    --adapter-path "${RETRIEVER_ADAPTER}" \
    --pooling mean \
    --top-n "${TOP_N}" \
    --output "experiment2/results/retrieved_top${TOP_N}.jsonl" \
    "${MAX_SAMPLES_ARG[@]}"

  python experiment2/rerank_clapnq.py \
    --retrieval-file "experiment2/results/retrieved_top${TOP_N}.jsonl" \
    --model-name "${RERANKER_MODEL}" \
    --top-n "${TOP_N}" \
    --output "experiment2/results/reranked_top${TOP_N}.jsonl" \
    "${MAX_SAMPLES_ARG[@]}"
fi

python experiment3/evaluate_generator.py \
  --questions-file "${DEV_FILE}" \
  --model-name "${GENERATOR_MODEL}" \
  --mode gold \
  --passages-file "${PASSAGES_FILE}" \
  --output experiment3/results/original_gold.jsonl \
  "${MAX_SAMPLES_ARG[@]}"

python experiment3/evaluate_generator.py \
  --questions-file "${DEV_FILE}" \
  --model-name "${GENERATOR_MODEL}" \
  --adapter-path "${GENERATOR_ADAPTER}" \
  --mode gold \
  --passages-file "${PASSAGES_FILE}" \
  --output experiment3/results/finetuned_gold.jsonl \
  "${MAX_SAMPLES_ARG[@]}"

python experiment3/evaluate_generator.py \
  --questions-file "${DEV_FILE}" \
  --model-name "${GENERATOR_MODEL}" \
  --mode reranked \
  --retrieval-file "experiment2/results/reranked_top${TOP_N}.jsonl" \
  --k "${BEST_K}" \
  --output "experiment3/results/original_rag_reranker_k${BEST_K}.jsonl" \
  "${MAX_SAMPLES_ARG[@]}"

python experiment3/evaluate_generator.py \
  --questions-file "${DEV_FILE}" \
  --model-name "${GENERATOR_MODEL}" \
  --adapter-path "${GENERATOR_ADAPTER}" \
  --mode reranked \
  --retrieval-file "experiment2/results/reranked_top${TOP_N}.jsonl" \
  --k "${BEST_K}" \
  --output "experiment3/results/finetuned_rag_reranker_k${BEST_K}.jsonl" \
  "${MAX_SAMPLES_ARG[@]}"

echo "===== Experiment 3 metrics ====="
cat experiment3/results/*.metrics.json

echo "===== Experiment 3 generator SFT job finished ====="
date
