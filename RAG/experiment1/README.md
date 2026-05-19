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

如果 Hugging Face 或镜像下载 BGE 不稳定，可以改用魔搭社区 ModelScope：

```bash
pip install modelscope
rm -rf models/bge-base-en-v1.5
mkdir -p models/bge-base-en-v1.5

modelscope download --model BAAI/bge-base-en-v1.5 \
  --local_dir models/bge-base-en-v1.5
```

如果上面的 ModelScope 模型 ID 不可用，使用镜像仓库 ID：

```bash
modelscope download --model AI-ModelScope/bge-base-en-v1.5 \
  --local_dir models/bge-base-en-v1.5
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
  --max-train all \
  --max-dev all \
  --max-train-positives all \
  --max-train-negatives all \
  --max-train-hard-negatives all
```

这里的 `all` 表示不截断样本，也不截断每条训练样本中的 positive、negative 和 hard negative passages。生成后检查行数：

```bash
wc -l experiment1/data_full/train.jsonl
wc -l experiment1/data_full/dev.jsonl
```

报告中应记录实际使用的训练样本数和评估样本数。

如果使用作业系统提交完整实验，可以直接提交脚本：

```bash
sbatch experiment1/slurm_exp1_debug.sh
```

这个脚本会使用全量 DPR-NQ 数据，并依次完成原始 BERT baseline、BERT + LoRA、BERT + LoRA + hard negatives、BGE 对比模型评估。

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
| BERT-base + mean pooling | 110M | 0.2973 | 0.5922 | 0.7332 | 0.8100 |
| BERT-base + LoRA | 110M | 0.4535 | 0.7501 | 0.8589 | 0.9071 |
| BERT-base + LoRA + hard negatives | 110M | 0.5221 | 0.8152 | 0.9042 | 0.9404 |
| BGE-base-en-v1.5 | 109M | 0.7605 | 0.9512 | 0.9785 | 0.9880 |

本次评估使用完整 DPR-NQ dev candidate passages，共 6515 个有效评估样本。结果显示，原始 BERT mean pooling 已经具备一定候选排序能力，但经过 LoRA 对比学习训练后，MRR@10 从 0.2973 提升到 0.4535，Recall@10 从 0.5922 提升到 0.7501。继续引入 hard negatives 后，MRR@10 进一步提升到 0.5221，Recall@10 提升到 0.8152，说明 hard negatives 能帮助模型区分表面相关但不能正确回答问题的 passages。

BGE-base-en-v1.5 在所有指标上仍明显领先，MRR@10 达到 0.7605，Recall@10 达到 0.9512。这说明成熟 embedding 模型经过更大规模、更系统的数据训练后，检索能力仍强于本实验中基于 DPR-NQ 训练 1 个 epoch 的 BERT retriever。

## 10. 提交作业时需要说明

实验报告中建议明确写出：

- 使用完整 DPR-NQ train 训练
- 使用完整 DPR-NQ dev 做 candidate passages 排序评估
- BERT pooling、normalization、LoRA 配置
- batch size、epoch、learning rate、temperature
- 是否使用 hard negatives
- BGE-base-en-v1.5 的对比结果
- 原始 BERT、训练后 BERT、hard negative 训练后 BERT、BGE 之间的差距分析
