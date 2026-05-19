#!/usr/bin/env bash
#SBATCH --job-name=rag-exp1-full
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --time=24:00:00

set -euo pipefail

echo "===== Experiment 1 full DPR-NQ job started ====="
date
hostname

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
echo "project root: $(pwd)"

nvidia-smi || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rag-exp

python --version
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

DATA_DIR="experiment1/data_full"
RESULT_DIR="experiment1/results"
CKPT_DIR="experiment1/checkpoints"
MODEL_DIR="models"
BERT_MODEL="${MODEL_DIR}/bert-base-uncased"
BGE_MODEL="${MODEL_DIR}/bge-base-en-v1.5"

BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-2e-5}"

mkdir -p "${DATA_DIR}" "${RESULT_DIR}" "${CKPT_DIR}" "${MODEL_DIR}"

download_model_if_needed() {
  local repo_id="$1"
  local local_dir="$2"

  if [[ -f "${local_dir}/config.json" && -f "${local_dir}/model.safetensors" ]]; then
    echo "model already exists: ${local_dir}"
    return
  fi

  if ! command -v hf >/dev/null 2>&1; then
    echo "hf command not found. Install huggingface_hub or put ${repo_id} under ${local_dir} manually." >&2
    exit 1
  fi

  echo "downloading ${repo_id} to ${local_dir}"
  hf download "${repo_id}" --local-dir "${local_dir}"
}

download_bge_from_modelscope_if_needed() {
  local local_dir="$1"

  if [[ -f "${local_dir}/config.json" && -f "${local_dir}/model.safetensors" ]]; then
    echo "BGE model already exists: ${local_dir}"
    return
  fi

  if ! command -v modelscope >/dev/null 2>&1; then
    echo "modelscope command not found; installing modelscope"
    pip install modelscope
  fi

  echo "downloading BGE from ModelScope to ${local_dir}"
  rm -rf "${local_dir}"
  mkdir -p "${local_dir}"

  if modelscope download --model BAAI/bge-base-en-v1.5 --local_dir "${local_dir}"; then
    return
  fi

  echo "BAAI/bge-base-en-v1.5 was not available from ModelScope; trying AI-ModelScope/bge-base-en-v1.5"
  modelscope download --model AI-ModelScope/bge-base-en-v1.5 --local_dir "${local_dir}"
}

download_model_if_needed "google-bert/bert-base-uncased" "${BERT_MODEL}"
download_bge_from_modelscope_if_needed "${BGE_MODEL}"

if [[ ! -f "${DATA_DIR}/train.jsonl" || ! -f "${DATA_DIR}/dev.jsonl" ]]; then
  echo "converting full DPR-NQ train/dev JSON files to JSONL"
  python experiment1/make_dpr_subset.py \
    --train-file DPR-NQ/biencoder-nq-train.json \
    --dev-file DPR-NQ/biencoder-nq-dev.json \
    --out-dir "${DATA_DIR}" \
    --max-train all \
    --max-dev all \
    --max-train-positives all \
    --max-train-negatives all \
    --max-train-hard-negatives all
else
  echo "full DPR-NQ JSONL files already exist under ${DATA_DIR}"
fi

wc -l "${DATA_DIR}/train.jsonl"
wc -l "${DATA_DIR}/dev.jsonl"

echo "===== Baseline: raw BERT mean pooling ====="
python experiment1/evaluate_dpr_candidates.py \
  --dev-file "${DATA_DIR}/dev.jsonl" \
  --model-name "${BERT_MODEL}" \
  --pooling mean \
  --output "${RESULT_DIR}/raw_bert_dev_full.json"

echo "===== Stage 1: BERT + LoRA with normal negatives ====="
python experiment1/train_bert_retriever.py \
  --train-file "${DATA_DIR}/train.jsonl" \
  --output-dir "${CKPT_DIR}/bert_lora_full" \
  --model-name "${BERT_MODEL}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --max-query-length 64 \
  --max-passage-length 256 \
  --fp16

echo "===== Evaluate: BERT + LoRA ====="
python experiment1/evaluate_dpr_candidates.py \
  --dev-file "${DATA_DIR}/dev.jsonl" \
  --model-name "${BERT_MODEL}" \
  --adapter-path "${CKPT_DIR}/bert_lora_full" \
  --pooling mean \
  --output "${RESULT_DIR}/bert_lora_dev_full.json"

echo "===== Stage 2: BERT + LoRA + hard negatives ====="
python experiment1/train_bert_retriever.py \
  --train-file "${DATA_DIR}/train.jsonl" \
  --output-dir "${CKPT_DIR}/bert_lora_hn_full" \
  --model-name "${BERT_MODEL}" \
  --adapter-path "${CKPT_DIR}/bert_lora_full" \
  --use-hard-negatives \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --max-query-length 64 \
  --max-passage-length 256 \
  --fp16

echo "===== Evaluate: BERT + LoRA + hard negatives ====="
python experiment1/evaluate_dpr_candidates.py \
  --dev-file "${DATA_DIR}/dev.jsonl" \
  --model-name "${BERT_MODEL}" \
  --adapter-path "${CKPT_DIR}/bert_lora_hn_full" \
  --pooling mean \
  --output "${RESULT_DIR}/bert_lora_hn_dev_full.json"

echo "===== Evaluate: BGE-base-en-v1.5 ====="
python experiment1/evaluate_dpr_candidates.py \
  --dev-file "${DATA_DIR}/dev.jsonl" \
  --model-name "${BGE_MODEL}" \
  --pooling cls \
  --output "${RESULT_DIR}/bge_base_en_v15_dev_full.json"

echo "===== Result files ====="
ls -lh "${RESULT_DIR}"/*_full.json
cat "${RESULT_DIR}/raw_bert_dev_full.json"
cat "${RESULT_DIR}/bert_lora_dev_full.json"
cat "${RESULT_DIR}/bert_lora_hn_dev_full.json"
cat "${RESULT_DIR}/bge_base_en_v15_dev_full.json"

echo "===== Experiment 1 full DPR-NQ job finished ====="
date
