# 实验一：BERT Dense Retriever

本目录包含 RAG 作业第一部分的代码，目标是训练并评估一个基于 BERT 的 dense retriever：

- 将 `google-bert/bert-base-uncased` 训练成 bi-encoder dense retriever
- 使用带 `attention_mask` 的 mean pooling，并对向量做 L2 normalization
- 使用 LoRA 和对比学习进行微调
- 在 DPR-NQ dev 的候选 passages 上评估
- 与 `BAAI/bge-base-en-v1.5` 做同参数量级对比

## 1. 环境配置

在服务器上可以创建如下环境：

```bash
conda create -n rag-exp python=3.10 -y
conda activate rag-exp
pip install -r experiment1/requirements.txt
```

如果服务器 CUDA 版本与默认 PyTorch 不匹配，可以根据实际 CUDA 版本单独安装对应的 PyTorch。

## 2. 生成小规模调试数据

DPR 完整训练文件很大，建议先转成 JSONL 子集做快速调试：

```bash
python experiment1/make_dpr_subset.py \
  --train-file DPR-NQ/biencoder-nq-train.json \
  --dev-file DPR-NQ/biencoder-nq-dev.json \
  --out-dir experiment1/data \
  --max-train 20000 \
  --max-dev 1000 \
  --max-train-negatives 8 \
  --max-train-hard-negatives 8
```

训练子集默认会做压缩：每条训练样本保留 1 个 positive passage、最多 8 个普通 negative passages、最多 8 个 hard negative passages。
dev 子集会保留原始候选 passages，这样计算 Recall 和 MRR 才有意义。
如果服务器时间和显存允许，可以适当增大 `--max-train` 和 `--max-dev`。

## 3. Baseline：原始 BERT mean pooling

```bash
python experiment1/evaluate_dpr_candidates.py \
  --dev-file experiment1/data/dev.jsonl \
  --model-name google-bert/bert-base-uncased \
  --pooling mean \
  --output experiment1/results/raw_bert_dev.json
```

## 4. 训练 BERT + LoRA

第一阶段使用普通 negative passages：

```bash
python experiment1/train_bert_retriever.py \
  --train-file experiment1/data/train.jsonl \
  --output-dir experiment1/checkpoints/bert_lora \
  --model-name google-bert/bert-base-uncased \
  --epochs 1 \
  --batch-size 16 \
  --lr 2e-5 \
  --max-query-length 64 \
  --max-passage-length 256
```

第二阶段在第一阶段 checkpoint 基础上继续训练，并使用 hard negative passages：

```bash
python experiment1/train_bert_retriever.py \
  --train-file experiment1/data/train.jsonl \
  --output-dir experiment1/checkpoints/bert_lora_hn \
  --model-name google-bert/bert-base-uncased \
  --adapter-path experiment1/checkpoints/bert_lora \
  --use-hard-negatives \
  --epochs 1 \
  --batch-size 16 \
  --lr 2e-5
```

## 5. 评估训练后的 retriever

```bash
python experiment1/evaluate_dpr_candidates.py \
  --dev-file experiment1/data/dev.jsonl \
  --model-name google-bert/bert-base-uncased \
  --adapter-path experiment1/checkpoints/bert_lora \
  --pooling mean \
  --output experiment1/results/bert_lora_dev.json

python experiment1/evaluate_dpr_candidates.py \
  --dev-file experiment1/data/dev.jsonl \
  --model-name google-bert/bert-base-uncased \
  --adapter-path experiment1/checkpoints/bert_lora_hn \
  --pooling mean \
  --output experiment1/results/bert_lora_hn_dev.json
```

## 6. 评估 BGE 对比模型

```bash
python experiment1/evaluate_dpr_candidates.py \
  --dev-file experiment1/data/dev.jsonl \
  --model-name BAAI/bge-base-en-v1.5 \
  --pooling cls \
  --output experiment1/results/bge_base_en_v15_dev.json
```

## 7. 实验报告表格

运行完成后，将 `experiment1/results/` 中生成的指标填入作业要求的表格：

| 方法 | 参数量 | MRR@10 | Recall@10 | Recall@20 | Recall@30 |
|---|---:|---:|---:|---:|---:|
| BERT-base + mean pooling | 110M | | | | |
| BERT-base + LoRA | 110M | | | | |
| BERT-base + LoRA + hard negatives | 110M | | | | |
| BGE-base-en-v1.5 | 109M | | | | |
