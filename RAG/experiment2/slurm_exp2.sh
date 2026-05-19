#!/usr/bin/env bash
#SBATCH --job-name=rag-exp2
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --time=24:00:00

set -euo pipefail

echo "===== Experiment 2 RAG job started ====="
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
DEV_FILE="CLAPNQ/dev/question_dev_answerable.tsv"
INDEX_DIR="experiment2/index"
RESULT_DIR="experiment2/results"
GEN_DIR="experiment2/generations"

BERT_MODEL="${BERT_MODEL:-models/bert-base-uncased}"
RETRIEVER_ADAPTER="${RETRIEVER_ADAPTER:-experiment1/checkpoints/bert_lora_hn_full}"
RERANKER_MODEL="${RERANKER_MODEL:-models/bge-reranker-base}"
GENERATOR_MODEL="${GENERATOR_MODEL:-models/Qwen2.5-1.5B-Instruct}"

TOP_N="${TOP_N:-30}"
BEST_K="${BEST_K:-3}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-128}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-64}"
RERANK_BATCH_SIZE="${RERANK_BATCH_SIZE:-32}"
MAX_SAMPLES_ARG=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  MAX_SAMPLES_ARG=(--max-samples "${MAX_SAMPLES}")
fi

mkdir -p "${INDEX_DIR}" "${RESULT_DIR}" "${GEN_DIR}" models

download_from_hf_if_needed() {
  local repo_id="$1"
  local local_dir="$2"
  if [[ -f "${local_dir}/config.json" && \( -f "${local_dir}/model.safetensors" || -f "${local_dir}/pytorch_model.bin" \) ]]; then
    echo "model already exists: ${local_dir}"
    return
  fi
  if ! command -v hf >/dev/null 2>&1; then
    echo "hf command not found; please install huggingface_hub or download ${repo_id} manually" >&2
    exit 1
  fi
  hf download "${repo_id}" --local-dir "${local_dir}"
}

download_reranker_if_needed() {
  if [[ -f "${RERANKER_MODEL}/config.json" && \( -f "${RERANKER_MODEL}/model.safetensors" || -f "${RERANKER_MODEL}/pytorch_model.bin" \) ]]; then
    echo "reranker already exists: ${RERANKER_MODEL}"
    return
  fi
  if ! command -v modelscope >/dev/null 2>&1; then
    pip install modelscope
  fi
  rm -rf "${RERANKER_MODEL}"
  mkdir -p "${RERANKER_MODEL}"
  if modelscope download --model BAAI/bge-reranker-base --local_dir "${RERANKER_MODEL}"; then
    return
  fi
  modelscope download --model AI-ModelScope/bge-reranker-base --local_dir "${RERANKER_MODEL}"
}

download_from_hf_if_needed "google-bert/bert-base-uncased" "${BERT_MODEL}"
download_reranker_if_needed

if [[ ! -f "${INDEX_DIR}/clapnq.faiss" || ! -f "${INDEX_DIR}/passages.jsonl" ]]; then
  python experiment2/build_faiss_index.py \
    --passages-file "${PASSAGES_FILE}" \
    --model-name "${BERT_MODEL}" \
    --adapter-path "${RETRIEVER_ADAPTER}" \
    --pooling mean \
    --output-index "${INDEX_DIR}/clapnq.faiss" \
    --output-passages "${INDEX_DIR}/passages.jsonl" \
    --batch-size "${ENCODE_BATCH_SIZE}"
fi

python experiment2/retrieve_clapnq.py \
  --questions-file "${DEV_FILE}" \
  --index-file "${INDEX_DIR}/clapnq.faiss" \
  --passage-metadata "${INDEX_DIR}/passages.jsonl" \
  --model-name "${BERT_MODEL}" \
  --adapter-path "${RETRIEVER_ADAPTER}" \
  --pooling mean \
  --top-n "${TOP_N}" \
  --batch-size "${QUERY_BATCH_SIZE}" \
  --output "${RESULT_DIR}/retrieved_top${TOP_N}.jsonl" \
  "${MAX_SAMPLES_ARG[@]}"

python experiment2/evaluate_retrieval.py \
  --input "${RESULT_DIR}/retrieved_top${TOP_N}.jsonl" \
  --output "${RESULT_DIR}/retrieved_top${TOP_N}_metrics.json"

python experiment2/rerank_clapnq.py \
  --retrieval-file "${RESULT_DIR}/retrieved_top${TOP_N}.jsonl" \
  --model-name "${RERANKER_MODEL}" \
  --top-n "${TOP_N}" \
  --batch-size "${RERANK_BATCH_SIZE}" \
  --output "${RESULT_DIR}/reranked_top${TOP_N}.jsonl" \
  "${MAX_SAMPLES_ARG[@]}"

python experiment2/evaluate_retrieval.py \
  --input "${RESULT_DIR}/reranked_top${TOP_N}.jsonl" \
  --output "${RESULT_DIR}/reranked_top${TOP_N}_metrics.json"

for K in 1 3 5 10; do
  python experiment2/evaluate_retrieval.py \
    --input "${RESULT_DIR}/reranked_top${TOP_N}.jsonl" \
    --ks "${K}" \
    --output "${RESULT_DIR}/reranked_k${K}_retrieval_metrics.json"
done

if [[ -d "${GENERATOR_MODEL}" ]]; then
  for K in 1 3 5 10; do
    python experiment2/generate_answers.py \
      --questions-file "${DEV_FILE}" \
      --model-name "${GENERATOR_MODEL}" \
      --mode reranked \
      --retrieval-file "${RESULT_DIR}/reranked_top${TOP_N}.jsonl" \
      --k "${K}" \
      --output "${GEN_DIR}/rag_reranker_k${K}.jsonl" \
      "${MAX_SAMPLES_ARG[@]}"

    python experiment2/summarize_generation.py \
      --input "${GEN_DIR}/rag_reranker_k${K}.jsonl" \
      --output "${GEN_DIR}/rag_reranker_k${K}_metrics.json"
  done

  python experiment2/generate_answers.py \
    --questions-file "${DEV_FILE}" \
    --model-name "${GENERATOR_MODEL}" \
    --mode no_rag \
    --output "${GEN_DIR}/no_rag.jsonl" \
    "${MAX_SAMPLES_ARG[@]}"

  python experiment2/generate_answers.py \
    --questions-file "${DEV_FILE}" \
    --model-name "${GENERATOR_MODEL}" \
    --mode random \
    --passages-file "${PASSAGES_FILE}" \
    --k "${BEST_K}" \
    --output "${GEN_DIR}/random_k${BEST_K}.jsonl" \
    "${MAX_SAMPLES_ARG[@]}"

  python experiment2/generate_answers.py \
    --questions-file "${DEV_FILE}" \
    --model-name "${GENERATOR_MODEL}" \
    --mode rag \
    --retrieval-file "${RESULT_DIR}/retrieved_top${TOP_N}.jsonl" \
    --k "${BEST_K}" \
    --output "${GEN_DIR}/rag_no_reranker_k${BEST_K}.jsonl" \
    "${MAX_SAMPLES_ARG[@]}"

  for file in "${GEN_DIR}"/*.jsonl; do
    python experiment2/summarize_generation.py \
      --input "${file}" \
      --output "${file%.jsonl}_metrics.json"
  done
else
  echo "Generator model directory not found: ${GENERATOR_MODEL}"
  echo "Download Qwen2.5-1.5B-Instruct first, then rerun generation commands."
fi

echo "===== Experiment 2 RAG job finished ====="
date
