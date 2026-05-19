# 实验二：完整 RAG 系统构建与评估

本目录对应作业实验二，目标是在 CLAPNQ corpus 上构建完整 RAG 系统，并比较不同检索设置对最终问答效果的影响。

实验二流程：

1. 使用实验一训练得到的 BERT retriever 编码 CLAPNQ passages
2. 构建 FAISS 向量索引
3. 对 CLAPNQ dev questions 检索 Top-N passages
4. 使用 `BAAI/bge-reranker-base` 对 Top-N passages 重排
5. 对 k=1,3,5,10 比较 RAG + reranker 的检索效果和生成效果
6. 固定最佳 k，比较 No RAG、Random-k、RAG without Reranker、RAG with Reranker

## 1. 环境配置

```bash
conda activate rag-exp
pip install -r experiment2/requirements.txt
```

如果已经安装过实验一依赖，只需要补装：

```bash
pip install faiss-cpu modelscope
```

## 2. 准备模型

实验二默认使用这些本地路径：

```text
models/bert-base-uncased
experiment1/checkpoints/bert_lora_hn_full
models/bge-reranker-base
models/Qwen2.5-1.5B-Instruct
```

BERT 和实验一 checkpoint 应该已经存在。reranker 可以用魔搭社区下载：

```bash
mkdir -p models/bge-reranker-base
modelscope download --model BAAI/bge-reranker-base \
  --local_dir models/bge-reranker-base
```

如果该 ID 不可用，改用：

```bash
modelscope download --model AI-ModelScope/bge-reranker-base \
  --local_dir models/bge-reranker-base
```

Qwen generator 可以用 Hugging Face 或魔搭社区下载到：

```text
models/Qwen2.5-1.5B-Instruct
```

## 3. 构建 CLAPNQ FAISS 索引

```bash
python experiment2/build_faiss_index.py \
  --passages-file CLAPNQ/passages/passages.tsv \
  --model-name models/bert-base-uncased \
  --adapter-path experiment1/checkpoints/bert_lora_hn_full \
  --pooling mean \
  --output-index experiment2/index/clapnq.faiss \
  --output-passages experiment2/index/passages.jsonl \
  --batch-size 128
```

## 4. Retriever 检索 Top-N

```bash
python experiment2/retrieve_clapnq.py \
  --questions-file CLAPNQ/dev/question_dev_answerable.tsv \
  --index-file experiment2/index/clapnq.faiss \
  --passage-metadata experiment2/index/passages.jsonl \
  --model-name models/bert-base-uncased \
  --adapter-path experiment1/checkpoints/bert_lora_hn_full \
  --pooling mean \
  --top-n 30 \
  --output experiment2/results/retrieved_top30.jsonl
```

评估 retriever 检索结果：

```bash
python experiment2/evaluate_retrieval.py \
  --input experiment2/results/retrieved_top30.jsonl \
  --output experiment2/results/retrieved_top30_metrics.json
```

## 5. Reranker 重排

```bash
python experiment2/rerank_clapnq.py \
  --retrieval-file experiment2/results/retrieved_top30.jsonl \
  --model-name models/bge-reranker-base \
  --top-n 30 \
  --output experiment2/results/reranked_top30.jsonl
```

评估 reranker 后的检索效果：

```bash
python experiment2/evaluate_retrieval.py \
  --input experiment2/results/reranked_top30.jsonl \
  --output experiment2/results/reranked_top30_metrics.json
```

## 6. 不同 k 的影响实验

检索指标：

```bash
for K in 1 3 5 10; do
  python experiment2/evaluate_retrieval.py \
    --input experiment2/results/reranked_top30.jsonl \
    --ks ${K} \
    --output experiment2/results/reranked_k${K}_retrieval_metrics.json
done
```

生成指标：

```bash
for K in 1 3 5 10; do
  python experiment2/generate_answers.py \
    --questions-file CLAPNQ/dev/question_dev_answerable.tsv \
    --model-name models/Qwen2.5-1.5B-Instruct \
    --mode reranked \
    --retrieval-file experiment2/results/reranked_top30.jsonl \
    --k ${K} \
    --output experiment2/generations/rag_reranker_k${K}.jsonl

  python experiment2/summarize_generation.py \
    --input experiment2/generations/rag_reranker_k${K}.jsonl \
    --output experiment2/generations/rag_reranker_k${K}_metrics.json
done
```

根据 F1 和 ROUGE-L 选择最佳 k。如果多个 k 接近，优先选择较小的 k。本次不同 k 实验中，k=5 时 F1 和 ROUGE-L 均达到最高，因此后续固定 k*=5。

## 7. 固定 k 的四组对比实验

本实验选择最佳 k 为 5：

```bash
BEST_K=5

python experiment2/generate_answers.py \
  --questions-file CLAPNQ/dev/question_dev_answerable.tsv \
  --model-name models/Qwen2.5-1.5B-Instruct \
  --mode no_rag \
  --output experiment2/generations/no_rag.jsonl

python experiment2/generate_answers.py \
  --questions-file CLAPNQ/dev/question_dev_answerable.tsv \
  --model-name models/Qwen2.5-1.5B-Instruct \
  --mode random \
  --passages-file CLAPNQ/passages/passages.tsv \
  --k ${BEST_K} \
  --output experiment2/generations/random_k${BEST_K}.jsonl

python experiment2/generate_answers.py \
  --questions-file CLAPNQ/dev/question_dev_answerable.tsv \
  --model-name models/Qwen2.5-1.5B-Instruct \
  --mode rag \
  --retrieval-file experiment2/results/retrieved_top30.jsonl \
  --k ${BEST_K} \
  --output experiment2/generations/rag_no_reranker_k${BEST_K}.jsonl

python experiment2/generate_answers.py \
  --questions-file CLAPNQ/dev/question_dev_answerable.tsv \
  --model-name models/Qwen2.5-1.5B-Instruct \
  --mode reranked \
  --retrieval-file experiment2/results/reranked_top30.jsonl \
  --k ${BEST_K} \
  --output experiment2/generations/rag_reranker_k${BEST_K}.jsonl
```

汇总生成指标：

```bash
for file in experiment2/generations/*.jsonl; do
  python experiment2/summarize_generation.py \
    --input "${file}" \
    --output "${file%.jsonl}_metrics.json"
done
```

## 8. 一键作业脚本

```bash
sbatch experiment2/slurm_exp2.sh
```

可以通过环境变量覆盖默认配置：

```bash
BEST_K=5 TOP_N=30 sbatch experiment2/slurm_exp2.sh
```

调试时只跑少量样本：

```bash
MAX_SAMPLES=20 BEST_K=5 sbatch experiment2/slurm_exp2.sh
```

## 9. 结果表

表 1：RAG + Reranker 下不同 k 的生成效果

| k | F1 | ROUGE-L |
|---:|---:|---:|
| 1 |0.2298|0.1316|
| 3 |0.2433 |0.1451 |
| 5 |0.2503 |0.1467 |
| 10 |0.2152 |0.1288 |

表 2：RAG + Reranker 下不同 k 的检索效果

| k | Recall@k | MRR@10 | nDCG@k |
|---:|---:|---:|---:|
| 1 | 0.44|0.5533 |0.44 |
| 3 | 0.6333|0.5533 |0.5537 |
| 5 |0.71 |0.5533 |0.5855 |
| 10 | 0.7767|0.5533 |0.6075 |

表 3：固定 k 后四组方法的生成效果

| 方法 | k | F1 | ROUGE-L |
|---|---:|---:|---:|
| No RAG | - |0.1747 |0.0889 |
| Random-k | 5 |0.1523 |0.0833 |
| RAG Top-k without Reranker | 5 |0.2251 |0.1324 |
| RAG Top-k with Reranker | 5 | 0.2503|0.1467 |

表 4：固定 k 后 RAG 方法的检索效果

| 方法 | k | Recall@k | MRR@10 | nDCG@k |
|---|---:|---:|---:|---:|
| RAG Top-k without Reranker | 5 |0.5033 |0.3027 |0.3394 |
| RAG Top-k with Reranker | 5 |0.71 | 0.5533|0.5855 |
