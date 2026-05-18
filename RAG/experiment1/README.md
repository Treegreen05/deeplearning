# 实验一：BERT Dense Retriever

本目录对应 RAG 作业的实验一，目标是使用完整 DPR-NQ 数据训练并评估一个基于 BERT 的 dense retriever。

实验一需要完成以下对比：

- 原始 `google-bert/bert-base-uncased` + mean pooling
- `google-bert/bert-base-uncased` + LoRA 检索训练
- `google-bert/bert-base-uncased` + LoRA + hard negatives
- `BAAI/bge-base-en-v1.5` 同参数量级对比

实现设置：

- 结构：bi-encoder dense retriever
- Pooling：带 `attention_mask` 的 mean pooling
- 向量归一化：L2 normalization
- 相似度：归一化向量 dot product，即 cosine similarity
- 微调方式：LoRA
- 损失函数：InfoNCE / Multiple Negatives Ranking Loss
- 评估方式：DPR-NQ dev candidate passages 排序评估

## 1. 环境配置

在服务器上创建并激活环境：

```bash
conda create -n rag-exp python=3.10 -y
conda activate rag-exp
pip install -r experiment1/requirements.txt
```

如果 `conda create` 提示 Anaconda Terms of Service 未接受，先运行：

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

检查 PyTorch 是否能使用 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## 2. 准备模型文件

如果服务器可以直接访问 Hugging Face，后续命令可以直接使用模型名：

```bash
google-bert/bert-base-uncased
BAAI/bge-base-en-v1.5
```

如果服务器无法访问 Hugging Face，可以先通过镜像下载到本地：

```bash
mkdir -p models
export HF_ENDPOINT=https://hf-mirror.com

hf download google-bert/bert-base-uncased \
  --local-dir models/bert-base-uncased

hf download BAAI/bge-base-en-v1.5 \
  --local-dir models/bge-base-en-v1.5
```

如果使用本地模型目录，后续命令中的 `--model-name` 改成：

```bash
models/bert-base-uncased
models/bge-base-en-v1.5
```

## 3. 生成全量 DPR-NQ JSONL 数据

本实验正式结果必须使用完整 DPR-NQ train 和完整 DPR-NQ dev。原始 DPR 文件是大的 JSON 数组，训练和评估前先转换成 JSONL。

从项目根目录 `RAG/` 执行：

```bash
python experiment1/make_dpr_subset.py \
  --train-file DPR-NQ/biencoder-nq-train.json \
  --dev-file DPR-NQ/biencoder-nq-dev.json \
  --out-dir experiment1/data_full \
  --max-train 100000000 \
  --max-dev 100000000 \
  --max-train-positives 100000000 \
  --max-train-negatives 100000000 \
  --max-train-hard-negatives 100000000
```

这里把上限设置成远大于 DPR-NQ 实际规模的数值，作用是“不截断样本”。生成后检查行数：

```bash
wc -l experiment1/data_full/train.jsonl
wc -l experiment1/data_full/dev.jsonl
```

报告中应记录实际使用的训练样本数和评估样本数。

## 4. Baseline：原始 BERT Mean Pooling

如果可以联网或模型已经在 Hugging Face 缓存中：

```bash
python experiment1/evaluate_dpr_candidates.py \
  --dev-file experiment1/data_full/dev.jsonl \
  --model-name google-bert/bert-base-uncased \
  --pooling mean \
  --output experiment1/results/raw_bert_dev_full.json
```

如果使用本地模型目录：

```bash
python experiment1/evaluate_dpr_candidates.py \
  --dev-file experiment1/data_full/dev.jsonl \
  --model-name models/bert-base-uncased \
  --pooling mean \
  --output experiment1/results/raw_bert_dev_full.json
```

## 5. 第一阶段训练：BERT + LoRA

第一阶段使用 positive passages、普通 negative passages 和 in-batch negatives 进行对比学习。

联网或已缓存模型：

```bash
python experiment1/train_bert_retriever.py \
  --train-file experiment1/data_full/train.jsonl \
  --output-dir experiment1/checkpoints/bert_lora_full \
  --model-name google-bert/bert-base-uncased \
  --epochs 1 \
  --batch-size 32 \
  --lr 2e-5 \
  --max-query-length 64 \
  --max-passage-length 256 \
  --fp16
```

使用本地模型目录：

```bash
python experiment1/train_bert_retriever.py \
  --train-file experiment1/data_full/train.jsonl \
  --output-dir experiment1/checkpoints/bert_lora_full \
  --model-name models/bert-base-uncased \
  --epochs 1 \
  --batch-size 32 \
  --lr 2e-5 \
  --max-query-length 64 \
  --max-passage-length 256 \
  --fp16
```

如果显存充足，可以把 `--batch-size` 增大到 64 或 128。为了和实验报告一致，最终使用的 batch size、epoch、learning rate 需要记录下来。

## 6. 第二阶段训练：BERT + LoRA + Hard Negatives

第二阶段从第一阶段 checkpoint 继续训练，改用 hard negative passages。

联网或已缓存模型：

```bash
python experiment1/train_bert_retriever.py \
  --train-file experiment1/data_full/train.jsonl \
  --output-dir experiment1/checkpoints/bert_lora_hn_full \
  --model-name google-bert/bert-base-uncased \
  --adapter-path experiment1/checkpoints/bert_lora_full \
  --use-hard-negatives \
  --epochs 1 \
  --batch-size 32 \
  --lr 2e-5 \
  --max-query-length 64 \
  --max-passage-length 256 \
  --fp16
```

使用本地模型目录：

```bash
python experiment1/train_bert_retriever.py \
  --train-file experiment1/data_full/train.jsonl \
  --output-dir experiment1/checkpoints/bert_lora_hn_full \
  --model-name models/bert-base-uncased \
  --adapter-path experiment1/checkpoints/bert_lora_full \
  --use-hard-negatives \
  --epochs 1 \
  --batch-size 32 \
  --lr 2e-5 \
  --max-query-length 64 \
  --max-passage-length 256 \
  --fp16
```

## 7. 评估训练后的 Retriever

评估第一阶段模型：

```bash
python experiment1/evaluate_dpr_candidates.py \
  --dev-file experiment1/data_full/dev.jsonl \
  --model-name models/bert-base-uncased \
  --adapter-path experiment1/checkpoints/bert_lora_full \
  --pooling mean \
  --output experiment1/results/bert_lora_dev_full.json
```

评估 hard negatives 第二阶段模型：

```bash
python experiment1/evaluate_dpr_candidates.py \
  --dev-file experiment1/data_full/dev.jsonl \
  --model-name models/bert-base-uncased \
  --adapter-path experiment1/checkpoints/bert_lora_hn_full \
  --pooling mean \
  --output experiment1/results/bert_lora_hn_dev_full.json
```

如果不使用本地模型目录，把上面命令中的 `models/bert-base-uncased` 换成 `google-bert/bert-base-uncased`。

## 8. 评估 BGE 对比模型

使用本地模型目录：

```bash
python experiment1/evaluate_dpr_candidates.py \
  --dev-file experiment1/data_full/dev.jsonl \
  --model-name models/bge-base-en-v1.5 \
  --pooling cls \
  --output experiment1/results/bge_base_en_v15_dev_full.json
```

如果可以直接访问 Hugging Face：

```bash
python experiment1/evaluate_dpr_candidates.py \
  --dev-file experiment1/data_full/dev.jsonl \
  --model-name BAAI/bge-base-en-v1.5 \
  --pooling cls \
  --output experiment1/results/bge_base_en_v15_dev_full.json
```

## 9. 结果文件

实验完成后，`experiment1/results/` 中应至少包含：

```text
raw_bert_dev_full.json
bert_lora_dev_full.json
bert_lora_hn_dev_full.json
bge_base_en_v15_dev_full.json
```

查看结果：

```bash
cat experiment1/results/raw_bert_dev_full.json
cat experiment1/results/bert_lora_dev_full.json
cat experiment1/results/bert_lora_hn_dev_full.json
cat experiment1/results/bge_base_en_v15_dev_full.json
```

将其中的指标填入实验一表格：

| 方法 | 参数量 | MRR@10 | Recall@10 | Recall@20 | Recall@30 |
|---|---:|---:|---:|---:|---:|
| BERT-base + mean pooling | 110M | | | | |
| BERT-base + LoRA | 110M | | | | |
| BERT-base + LoRA + hard negatives | 110M | | | | |
| BGE-base-en-v1.5 | 109M | | | | |

## 10. 提交作业时需要说明

实验报告中建议明确写出：

- 使用完整 DPR-NQ train 训练
- 使用完整 DPR-NQ dev 做 candidate passages 排序评估
- BERT pooling、normalization、LoRA 配置
- batch size、epoch、learning rate、temperature
- 是否使用 hard negatives
- BGE-base-en-v1.5 的对比结果
- 原始 BERT、训练后 BERT、hard negative 训练后 BERT、BGE 之间的差距分析
