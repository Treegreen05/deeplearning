# 实验三：Generator 微调与端到端 RAG 效果评估

本目录对应作业实验三，目标是使用 CLAPNQ train 对 `Qwen2.5-1.5B-Instruct` 进行 LoRA 微调，并在 CLAPNQ dev 上比较微调前后的生成效果。

实验三延续实验二设置：

- Retriever：实验一训练得到的 BERT dense retriever
- Reranker：`BAAI/bge-reranker-base`
- Generator：`Qwen/Qwen2.5-1.5B-Instruct`
- Top-N：30
- 固定 k：5

注意：训练 generator 时，输入 passages 来自 RAG 检索和 reranker 重排结果，而不是直接使用 train 集中的 gold passage。

## 1. 环境配置

```bash
conda activate rag-exp
pip install -r experiment3/requirements.txt
```

如果实验二依赖已经安装好，通常只需要确认 `peft`、`transformers`、`modelscope` 可用。

## 2. 准备模型和检索结果

默认路径：

```text
models/Qwen2.5-1.5B-Instruct
models/bert-base-uncased
experiment1/checkpoints/bert_lora_hn_full
models/bge-reranker-base
experiment2/index/clapnq.faiss
experiment2/index/passages.jsonl
experiment2/results/reranked_top30.jsonl
```

如果 Qwen 尚未下载，可以使用魔搭社区：

```bash
mkdir -p models/Qwen2.5-1.5B-Instruct
modelscope download --model Qwen/Qwen2.5-1.5B-Instruct \
  --local_dir models/Qwen2.5-1.5B-Instruct
```

## 3. 为 CLAPNQ Train 检索训练 passages

先用实验二的 FAISS index 对 CLAPNQ train 检索 Top-30：

```bash
python experiment2/retrieve_clapnq.py \
  --questions-file CLAPNQ/train/question_train_answerable.tsv \
  --index-file experiment2/index/clapnq.faiss \
  --passage-metadata experiment2/index/passages.jsonl \
  --model-name models/bert-base-uncased \
  --adapter-path experiment1/checkpoints/bert_lora_hn_full \
  --pooling mean \
  --top-n 30 \
  --output experiment3/data/train_retrieved_top30.jsonl
```

再用 reranker 重排：

```bash
python experiment2/rerank_clapnq.py \
  --retrieval-file experiment3/data/train_retrieved_top30.jsonl \
  --model-name models/bge-reranker-base \
  --top-n 30 \
  --output experiment3/data/train_reranked_top30.jsonl
```

## 4. 构造 Generator SFT 数据

使用 k=5 的 reranked passages 作为 generator 输入，reference long answer 作为输出：

```bash
python experiment3/prepare_sft_data.py \
  --retrieval-file experiment3/data/train_reranked_top30.jsonl \
  --k 5 \
  --output experiment3/data/qwen_sft_train_k5.jsonl
```

输出 JSONL 中每行包含：

```text
prompt
answer
question
passages
```

## 5. LoRA 微调 Qwen Generator

```bash
python experiment3/train_qwen_lora.py \
  --train-file experiment3/data/qwen_sft_train_k5.jsonl \
  --model-name models/Qwen2.5-1.5B-Instruct \
  --output-dir experiment3/checkpoints/qwen_lora \
  --epochs 3 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --lr 2e-5 \
  --max-input-length 1536 \
  --max-output-length 256 \
  --fp16 \
  --gradient-checkpointing
```

如果显存充足，可以适当增大 `--batch-size` 或减少 `--grad-accum-steps`。

## 6. 微调前后评估

实验三需要比较四组：

1. 原始 Generator，gold passage
2. 微调 Generator，gold passage
3. 原始 Generator，RAG with Reranker
4. 微调 Generator，RAG with Reranker

### 6.1 Gold Passage 条件

原始 Generator：

```bash
python experiment3/evaluate_generator.py \
  --questions-file CLAPNQ/dev/question_dev_answerable.tsv \
  --model-name models/Qwen2.5-1.5B-Instruct \
  --mode gold \
  --passages-file CLAPNQ/passages/passages.tsv \
  --output experiment3/results/original_gold.jsonl
```

微调 Generator：

```bash
python experiment3/evaluate_generator.py \
  --questions-file CLAPNQ/dev/question_dev_answerable.tsv \
  --model-name models/Qwen2.5-1.5B-Instruct \
  --adapter-path experiment3/checkpoints/qwen_lora \
  --mode gold \
  --passages-file CLAPNQ/passages/passages.tsv \
  --output experiment3/results/finetuned_gold.jsonl
```

### 6.2 RAG with Reranker 条件

原始 Generator：

```bash
python experiment3/evaluate_generator.py \
  --questions-file CLAPNQ/dev/question_dev_answerable.tsv \
  --model-name models/Qwen2.5-1.5B-Instruct \
  --mode reranked \
  --retrieval-file experiment2/results/reranked_top30.jsonl \
  --k 5 \
  --output experiment3/results/original_rag_reranker_k5.jsonl
```

微调 Generator：

```bash
python experiment3/evaluate_generator.py \
  --questions-file CLAPNQ/dev/question_dev_answerable.tsv \
  --model-name models/Qwen2.5-1.5B-Instruct \
  --adapter-path experiment3/checkpoints/qwen_lora \
  --mode reranked \
  --retrieval-file experiment2/results/reranked_top30.jsonl \
  --k 5 \
  --output experiment3/results/finetuned_rag_reranker_k5.jsonl
```

每次评估会生成两个文件：

```text
*.jsonl
*.metrics.json
```

其中 `*.metrics.json` 可直接用于填写表格。

## 7. 一键脚本

如果需要一次性运行实验三：

```bash
sbatch experiment3/slurm_exp3.sh
```

调试少量样本：

```bash
MAX_SAMPLES=20 sbatch experiment3/slurm_exp3.sh
```

正式实验不要设置 `MAX_SAMPLES`。

## 8. 结果表

| 方法 | k | F1 | ROUGE-L |
|---|---:|---:|---:|
| 原始 Generator，gold passage | - | | |
| 微调 Generator，gold passage | - | | |
| 原始 Generator，RAG with Reranker | 5 | | |
| 微调 Generator，RAG with Reranker | 5 | | |

需要分析：

- 微调 generator 是否提升 gold passage 条件下的 F1 和 ROUGE-L
- 微调 generator 是否提升完整 RAG with reranker 条件下的端到端效果
- 微调后的 generator 是否更依赖给定 passages，而不是只凭模型内部知识回答
